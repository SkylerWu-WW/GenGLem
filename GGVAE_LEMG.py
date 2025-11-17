import os
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
import dgl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import DBSCAN
import logging
from datetime import datetime
from config import config
import random
import dgl.nn as dglnn
import networkx as nx
import torch
from sklearn.neighbors import KDTree





# 设置随机种子
seed = config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

node_dir = config["node_dir"]
edge_dir = config["edge_dir"]
node_output = config["node_output"]


def setup_logging(batch_name):
    model_output = config["model_output"]
    os.makedirs(model_output, exist_ok=True)
    log_file = os.path.join(model_output, f"{batch_name}_training.log")
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Started training for batch: {batch_name}")


class MoleculeDataset(Dataset):
    def __init__(self, node_dir, edge_dir):
        """
        node_dir: 存放节点信息的目录
        edge_dir: 存放边信息的目录
        """
        self.node_dir = node_dir
        self.edge_dir = edge_dir
        self.node_files = glob.glob(os.path.join(node_dir, "*_atom_numbers.npy"))
        self.node_files.sort()
        self.node_output = node_output
        # 创建一个字典来存储匹配的文件
        self.matched_files = {}
        for file in self.node_files:
            base_name = os.path.basename(file)
            molecule_frame = base_name.replace("_atom_numbers.npy", "")
            coordinates_file = os.path.join(node_dir, f"{molecule_frame}_coordinates.npy")
            if os.path.exists(coordinates_file):
                self.matched_files[molecule_frame] = (file, coordinates_file)
        # 缓存排序结果
        self.sorted_keys = self.get_sorted_keys()
        self._compute_global_normalization_params()


    def _compute_global_normalization_params(self):
        # 收集所有样本的坐标（shape: (N, 3)）到一个大数组中
        all_coords = []
        for molecule_frame, (_, coordinates_file) in self.matched_files.items():
            coords = np.load(coordinates_file)
            all_coords.append(coords)
        if all_coords:
            all_coords = np.concatenate(all_coords, axis=0)
            self.global_mean = torch.tensor(all_coords.mean(axis=0, keepdims=True), dtype=torch.float32)
            self.global_std = torch.tensor(all_coords.std(axis=0, keepdims=True) + 1e-6, dtype=torch.float32)
        else:
            self.global_mean = torch.zeros((1, 3), dtype=torch.float32)
            self.global_std = torch.ones((1, 3), dtype=torch.float32)

    def get_sorted_keys(self):
        def sort_key(molecule_frame):
            parts = molecule_frame.split("_")
            first_no = int(parts[1])
            second_no = int(parts[3])
            return (first_no, second_no)

        return sorted(self.matched_files.keys(), key=sort_key)

    def __len__(self):
        return len(self.matched_files)

    def __getitem__(self, idx):
        molecule_frame = self.sorted_keys[idx]
        atom_numbers_file, coordinates_file = self.matched_files[molecule_frame]
        # 加载原子序数 (N,)
        atom_numbers = np.load(atom_numbers_file)
        # 加载坐标信息 (N,3)
        coordinates = np.load(coordinates_file)
        # 构造节点特征：拼接原子序数和坐标，形状为 (N,4)
        node_features = np.concatenate([atom_numbers.reshape(-1, 1), coordinates], axis=-1)
        node_features = torch.tensor(node_features, dtype=torch.float32)
        # 对坐标部分（columns 1~3）进行全局归一化
        coords = node_features[:, 1:]
        norm_coords = (coords - self.global_mean) / self.global_std
        node_features[:, 1:] = norm_coords
        # 构造图；边特征部分保持原样
        bond_length_file = os.path.join(self.edge_dir, f"{molecule_frame}.xyz_bond_lengths.npy")
        if os.path.exists(bond_length_file):
            bonds = np.load(bond_length_file)
            if bonds.size != 0:
                bonds = bonds.astype(np.float32)
                src = bonds[:, 0].astype(np.int64)
                dst = bonds[:, 1].astype(np.int64)
                g = dgl.graph((src, dst), num_nodes=node_features.shape[0])
                edge_length_feat = torch.tensor(bonds[:, 2:], dtype=torch.float32)
                folder = molecule_frame.split("_frame")[0]
                bond_type_file = os.path.join(self.edge_dir, f"{folder}_bond_types.npy")
                if os.path.exists(bond_type_file):
                    bond_type_feat = torch.tensor(np.load(bond_type_file, allow_pickle=True), dtype=torch.float32)
                else:
                    bond_type_feat = None
                bond_aromaticity_file = os.path.join(self.edge_dir, f"{folder}_bond_aromaticity.npy")
                if os.path.exists(bond_aromaticity_file):
                    bond_aromaticity_feat = torch.tensor(np.load(bond_aromaticity_file, allow_pickle=True),
                                                         dtype=torch.float32)
                    if bond_aromaticity_feat.dim() == 1:
                        bond_aromaticity_feat = bond_aromaticity_feat.unsqueeze(1)
                else:
                    bond_aromaticity_feat = None
                bond_conjugation_file = os.path.join(self.edge_dir, f"{folder}_bond_conjugation.npy")
                if os.path.exists(bond_conjugation_file):
                    bond_conjugation_feat = torch.tensor(np.load(bond_conjugation_file, allow_pickle=True),
                                                         dtype=torch.float32)
                    if bond_conjugation_feat.dim() == 1:
                        bond_conjugation_feat = bond_conjugation_feat.unsqueeze(1)
                else:
                    bond_conjugation_feat = None
                bond_in_ring_file = os.path.join(self.edge_dir, f"{folder}_bond_in_ring.npy")
                if os.path.exists(bond_in_ring_file):
                    bond_in_ring_feat = torch.tensor(np.load(bond_in_ring_file, allow_pickle=True), dtype=torch.float32)
                else:
                    bond_in_ring_feat = None
                feature_list = [edge_length_feat]
                if bond_type_feat is not None:
                    feature_list.append(bond_type_feat)
                if bond_aromaticity_feat is not None:
                    feature_list.append(bond_aromaticity_feat)
                if bond_conjugation_feat is not None:
                    feature_list.append(bond_conjugation_feat)
                if bond_in_ring_feat is not None:
                    feature_list.append(bond_in_ring_feat)
                if len(feature_list) > 1:
                    edge_feat = torch.cat(feature_list, dim=1)
                else:
                    edge_feat = edge_length_feat
            else:
                g = dgl.graph(([], []), num_nodes=node_features.shape[0])
                edge_feat = torch.zeros((0, 1))
        else:
            g = dgl.graph(([], []), num_nodes=node_features.shape[0])
            edge_feat = torch.zeros((0, 1))
        # 为后续芳香环提取，保存芳香性标记（这里判断数值大于0.5为芳香键）
        if 'bond_aromaticity_feat' in locals() and bond_aromaticity_feat is not None:
            g.edata['aromatic'] = (bond_aromaticity_feat.squeeze() > 0.5)
        # 添加自环前，记录原始边数量
        original_edge_count = g.num_edges()
        # 添加自环，确保每个节点至少有一个入边
        g = dgl.add_self_loop(g)
        # 计算新增自环边的数量
        new_edge_count = g.num_edges() - original_edge_count
        # 假设自环边的特征维度与原边一致，初始化新增自环边为0
        self_loop_feat = torch.zeros((new_edge_count, edge_feat.shape[1]), dtype=edge_feat.dtype)
        # 拼接原始边特征和自环边的特征，
        new_edge_feat = torch.cat([edge_feat, self_loop_feat], dim=0)
        # 保存原始特征作为条件信息
        g.ndata['raw_h'] = node_features
        # 复制一份用于后续图卷积
        g.ndata['h'] = node_features.clone()
        g.edata['w'] = new_edge_feat
        g.molecule_frame = molecule_frame
        return g

    def save_node_features(self):
        for molecule_frame, (atom_numbers_file, coordinates_file) in self.matched_files.items():
            atom_numbers = np.load(atom_numbers_file)
            coordinates = np.load(coordinates_file)
            node_features = np.concatenate([atom_numbers.reshape(-1, 1), coordinates], axis=-1)
            output_file = os.path.join(self.node_output, f"{molecule_frame}_node_features.npy")
            np.save(output_file, node_features)
            print(f"Saved node features to {output_file}")

    def split_batches(self):
        batches = {}
        for idx, molecule_frame in enumerate(self.sorted_keys):
            batch_name = molecule_frame.split("_frame")[0]
            g = self.__getitem__(idx)
            if batch_name not in batches:
                batches[batch_name] = []
            batches[batch_name].append(g)
        return batches


def extract_aromatic_rings_from_arom_file(g, edge_dir):
    """
    利用图 g 中存储的芳香性标记（g.edata['aromatic']），转为 networkx 图后，
    仅保留芳香边，然后利用 cycle_basis 提取所有基本环（不限制环长）。
    对于融合环，可以后续进行合并处理；这里先直接返回所有基本环。
    """
    # 转换为 networkx 图，并保留 'aromatic' 属性
    nx_g = g.to_networkx(node_attrs=['raw_h'], edge_attrs=['aromatic'])
    # 手动移除自环：遍历所有边，删除起始和终点相同的边
    selfloops = [(u, v) for u, v in nx_g.edges() if u == v]
    nx_g.remove_edges_from(selfloops)
    # 仅保留芳香边：边上属性 aromatic 为 True
    aromatic_edges = [(u, v) for u, v, d in nx_g.edges(data=True) if d.get('aromatic', False)]
    G_arom = nx.Graph()
    G_arom.add_edges_from(aromatic_edges)
    cycles = nx.cycle_basis(G_arom)
    # 如果需要，你可以在这里添加对环长度的过滤条件，例如过滤掉小于5的环
    aromatic_cycles = [cycle for cycle in cycles if len(cycle) >= 5]
    return aromatic_cycles

def fit_plane(points):
    """
    对点集 (n,3) 利用 SVD 拟合平面，返回质心和法向量。
    如果所有点均包含 NaN，则返回默认值（如全零向量）。
    """
    # points 应该保持为 torch.Tensor，不要转为 numpy array，删除nan点的行
    valid_points = points[~torch.isnan(points).any(dim=1)]
    if valid_points.shape[0] == 0:
        print("Warning: 所有点均包含 NaN，返回默认值")
        default_centroid = torch.zeros(points.shape[1], device=points.device)
        default_normal = torch.zeros(points.shape[1], device=points.device)
        return default_centroid, default_normal
    centroid = torch.mean(valid_points, dim=0)
    centered_points = valid_points - centroid
    print("centered_points:", centered_points)  # 调试用，打印centered_points
    # 使用 torch.linalg.svd 进行 SVD 分解，保留梯度
    U, S, Vh = torch.linalg.svd(centered_points, full_matrices=False)
    # Vh 的最后一行就是最小奇异值对应的方向，也就是法向量
    normal = Vh[-1]
    return centroid, normal

def compute_planarity_loss(ring_coords):
    """
    计算各点到拟合平面的距离均方误差，确保操作在梯度路径中。
    ring_coords: Tensor, shape (n,3)
    """
    centroid, normal = fit_plane(ring_coords)
    # 计算距离：绝对值的点积 (point - centroid)·normal
    distances = torch.abs(torch.matmul(ring_coords - centroid, normal))
    return torch.mean(distances ** 2)

def compute_aromatic_planarity_loss(g, recon_coords, edge_dir):
    """
    对单个图 g，根据重构后的节点坐标 recon_coords (shape: (n,3))，
    利用芳香性信息提取所有芳香环，然后对每个环计算平面性 loss（均方误差），
    返回所有环的平均值。该 loss 计算确保在梯度路径中，可参与反向传播。
    """
    cycles = extract_aromatic_rings_from_arom_file(g, edge_dir)
    if len(cycles) == 0:
        return torch.tensor(0.0, device=recon_coords.device)
    loss_total = 0.0
    count = 0
    for cycle in cycles:
        if len(cycle) < 3:
            continue
        # 提取该芳香环内各节点的坐标（保持为 torch.Tensor）
        ring_coords = recon_coords[cycle, :]
        loss_total += compute_planarity_loss(ring_coords)
        count += 1
    if count > 0:
        return loss_total / count
    else:
        return torch.tensor(0.0, device=recon_coords.device)



def extract_bond_indices(config, edge_dir):
    # 加载预处理存储的bond_lengths文件
    bond_lengths_path = os.path.join(edge_dir, f"{config['batch_name']}_frame_0.xyz_bond_lengths.npy")
    if os.path.exists(bond_lengths_path):
        bonds = np.load(bond_lengths_path)
        # 假设bonds的前两列为原子对下标
        bond_indices = torch.tensor(bonds[:, :2], dtype=torch.long)
        return bond_indices
    else:
        return None

def extract_angle_indices(config, edge_dir):
    # 加载预处理存储的bond_angles文件
    bond_angles_path = os.path.join(edge_dir, f"{config['batch_name']}_frame_0.xyz_bond_angles.npy")
    if os.path.exists(bond_angles_path):
        angles = np.load(bond_angles_path)
        # 假设angles的前三列为角的原子下标
        angle_indices = torch.tensor(angles[:, :3], dtype=torch.long)
        return angle_indices
    else:
        return None


def extract_nonbonded_pairs(config, recon, edge_dir):
    """
    利用 KDTree 对原子坐标进行近邻搜索，仅返回距离小于 nonbond_threshold 的非键原子对。
    同时如果存在键的文件，则过滤掉已经存在键连接的原子对。
    """
    threshold = config["nonbond_threshold"]
    # recon 是重构后的坐标 (N,3) 的 torch.Tensor，转换为 numpy 数组
    coords = recon.cpu().detach().numpy()  # (N,3)
    tree = KDTree(coords)
    neighbors = tree.query_radius(coords, r=threshold)
    candidate_pairs = []
    N = coords.shape[0]
    for i in range(N):
        # 对于每个原子 i，仅考虑索引大于 i 的邻居，避免重复对称计算
        for j in neighbors[i]:
            if j > i:
                candidate_pairs.append([i, j])
    candidate_pairs = np.array(candidate_pairs)

    # 如果存在键文件，则过滤掉键对（假设键文件中记录的是原子对索引）
    bond_lengths_path = os.path.join(edge_dir, f"{config['batch_name']}_frame_0.xyz_bond_lengths.npy")
    if os.path.exists(bond_lengths_path):
        bonds = np.load(bond_lengths_path)
        bonded_pairs = set()
        for pair in bonds[:, :2]:
            i, j = int(pair[0]), int(pair[1])
            bonded_pairs.add((i, j))
            bonded_pairs.add((j, i))
        filtered_pairs = []
        for pair in candidate_pairs:
            if tuple(pair) not in bonded_pairs:
                filtered_pairs.append(pair)
        if len(filtered_pairs) > 0:
            candidate_pairs = np.array(filtered_pairs)
        else:
            return None
    if candidate_pairs.shape[0] > 0:
        return torch.tensor(candidate_pairs, dtype=torch.long, device=recon.device)
    else:
        return None


# 键长惩罚 loss 函数，基于原始坐标与重构坐标计算键长损失
def compute_modified_bond_length_loss(g, recon_coords, bond_indices):
    orig_coords = g.ndata['raw_h'][:, 1:4]
    atom1_orig = orig_coords[bond_indices[:, 0]]
    atom2_orig = orig_coords[bond_indices[:, 1]]
    orig_lengths = torch.norm(atom1_orig - atom2_orig, dim=1)
    atom1_pred = recon_coords[bond_indices[:, 0]]
    atom2_pred = recon_coords[bond_indices[:, 1]]
    pred_lengths = torch.norm(atom1_pred - atom2_pred, dim=1)
    loss_bond = torch.mean((pred_lengths - orig_lengths)**2)
    return loss_bond

# 键角惩罚 loss 函数，基于原始角度与重构角度计算键角损失
def compute_modified_bond_angle_loss(g, recon_coords, angle_indices):
    orig_coords = g.ndata['raw_h'][:, 1:4]
    # 提取角中三个原子的坐标
    atom_i_orig = orig_coords[angle_indices[:, 0]]
    atom_j_orig = orig_coords[angle_indices[:, 1]]
    atom_k_orig = orig_coords[angle_indices[:, 2]]
    vec1_orig = atom_i_orig - atom_j_orig
    vec2_orig = atom_k_orig - atom_j_orig
    cos_orig = F.cosine_similarity(vec1_orig, vec2_orig, dim=1)
    orig_angles = torch.acos(torch.clamp(cos_orig, -1.0 + 1e-6, 1.0 - 1e-6))

    atom_i_pred = recon_coords[angle_indices[:, 0]]
    atom_j_pred = recon_coords[angle_indices[:, 1]]
    atom_k_pred = recon_coords[angle_indices[:, 2]]
    vec1_pred = atom_i_pred - atom_j_pred
    vec2_pred = atom_k_pred - atom_j_pred
    cos_pred = F.cosine_similarity(vec1_pred, vec2_pred, dim=1)
    pred_angles = torch.acos(torch.clamp(cos_pred, -1.0 + 1e-6, 1.0 - 1e-6))

    loss_angle = torch.mean((pred_angles - orig_angles) ** 2)
    return loss_angle

# 非键相互作用（立体障碍）惩罚 loss 函数，基于原始非键对距离与重构非键对距离计算非键损失
def compute_modified_nonbond_loss(g, recon_coords, nonbonded_pairs):
    orig_coords = g.ndata['raw_h'][:, 1:4]
    atom1_orig = orig_coords[nonbonded_pairs[:, 0]]
    atom2_orig = orig_coords[nonbonded_pairs[:, 1]]
    orig_dists = torch.norm(atom1_orig - atom2_orig, dim=1)
    atom1_pred = recon_coords[nonbonded_pairs[:, 0]]
    atom2_pred = recon_coords[nonbonded_pairs[:, 1]]
    pred_dists = torch.norm(atom1_pred - atom2_pred, dim=1)
    loss_nonbond = torch.mean((pred_dists - orig_dists)**2)
    return loss_nonbond



class ConditionFusion(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(ConditionFusion, self).__init__()
        # 用于计算注意力得分的全连接层
        self.attn_fc = nn.Linear(feature_dim, hidden_dim)
        # 学习一个 context vector 用于内积打分
        self.context_vector = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, cond_feats):
        """
        cond_feats: Tensor, shape (num_graphs, N, feature_dim)
        输出融合后的条件特征: shape (N, feature_dim)
        """
        # 通过全连接映射
        attn_scores = torch.tanh(self.attn_fc(cond_feats))  # (num_graphs, N, hidden_dim)
        # 与 context vector 内积，得到得分 (num_graphs, N)
        attn_scores = torch.matmul(attn_scores, self.context_vector)  # (num_graphs, N)
        # 在图维度上归一化得到注意力权重
        attn_weights = torch.softmax(attn_scores, dim=0)  # (num_graphs, N)
        # 加权求和融合所有图的条件特征
        fused_cond_feat = torch.sum(attn_weights.unsqueeze(-1) * cond_feats, dim=0)  # (N, feature_dim)
        return fused_cond_feat


class GraphVAE(nn.Module):
    def __init__(self, in_feats, hidden_dim, z_dim, num_nodes):
        """
        这里的 num_nodes 表示每个 graph 的节点数（假设同一批次中所有图节点数一致）
        """
        super(GraphVAE, self).__init__()
        # 编码器部分
        self.conv1 = dgl.nn.GraphConv(in_feats, hidden_dim, activation=F.relu)
        self.conv2 = dgl.nn.GraphConv(hidden_dim, hidden_dim, activation=F.relu)
        self.conv3 = dgl.nn.GraphConv(hidden_dim, hidden_dim, activation=F.relu)
        self.conv4 = dgl.nn.GraphConv(hidden_dim, hidden_dim, activation=F.relu)
        self.fc_mu = nn.Linear(num_nodes * hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(num_nodes * hidden_dim, z_dim)
        self.fc_decode = nn.Linear(z_dim, num_nodes * hidden_dim)
        # 解码器部分
        self.gconv1 = dgl.nn.GraphConv(hidden_dim + in_feats + 2 * hidden_dim, hidden_dim, activation=F.relu)
        self.gconv2 = dgl.nn.GraphConv(hidden_dim, hidden_dim, activation=F.relu)
        self.fc_out = nn.Linear(hidden_dim, in_feats)

    def batched_encode(self, batched_graph):
        raw = batched_graph.ndata['raw_h']
        h1 = self.conv1(batched_graph, raw)
        h2 = self.conv2(batched_graph, h1)
        h3 = self.conv3(batched_graph, h2)
        h4 = self.conv4(batched_graph, h3)
        batched_graph.ndata['skip_multi'] = torch.cat([h1, h3], dim=1)
        batched_graph.ndata['h'] = h4
        node_counts = batched_graph.batch_num_nodes()
        n = node_counts[0]
        h_split = torch.split(batched_graph.ndata['h'], n)
        h_flat_list = [h_i.view(1, -1) for h_i in h_split]
        h_flat = torch.cat(h_flat_list, dim=0)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_batch(self, batched_graph):
        mu, logvar = self.batched_encode(batched_graph)
        z = self.reparameterize(mu, logvar)
        node_counts = batched_graph.batch_num_nodes()
        n = node_counts[0]
        num_graphs = len(node_counts)
        latent_list = []
        for i in range(num_graphs):
            latent_flat_i = self.fc_decode(z[i:i + 1])
            latent_node_i = latent_flat_i.view(n, -1)
            latent_list.append(latent_node_i)
        latent_node = torch.cat(latent_list, dim=0)
        cond_feat = batched_graph.ndata['raw_h']
        skip_feat = batched_graph.ndata['skip_multi']
        cond_feat = torch.cat([cond_feat, skip_feat], dim=1)
        decoder_input = torch.cat([latent_node, cond_feat], dim=1)
        x = self.gconv1(batched_graph, decoder_input)
        x = self.gconv2(batched_graph, x)
        recon = self.fc_out(x)
        # 手动将重构结果第一列设置为原始原子序数
        atomic_numbers = batched_graph.ndata['raw_h'][:, 0:1]  # 取出第一列
        recon[:, 0:1] = atomic_numbers
        return recon, mu, logvar

    def forward(self, g):
        mu, logvar = self.encode(g)
        z = self.reparameterize(mu, logvar)
        latent_flat = self.fc_decode(z)
        node_count = g.number_of_nodes()
        latent_node = latent_flat.view(node_count, -1)
        cond_feat = torch.cat([g.ndata['raw_h'], g.ndata['skip_multi']], dim=1)
        decoder_input = torch.cat([latent_node, cond_feat], dim=1)
        x = self.gconv1(g, decoder_input)
        x = self.gconv2(g, x)
        recon = self.fc_out(x)
        # 将重构结果第一列强制变为原始原子序数
        atomic_numbers = g.ndata['raw_h'][:, 0:1]
        recon[:, 0:1] = atomic_numbers
        return recon, mu, logvar

    def encode(self, g):
        raw = g.ndata['raw_h']
        h1 = self.conv1(g, raw)
        h2 = self.conv2(g, h1)
        h3 = self.conv3(g, h2)
        h4 = self.conv4(g, h3)
        g.ndata['skip_multi'] = torch.cat([h1, h3], dim=1)
        g.ndata['h'] = h4
        h_flat = g.ndata['h'].view(1, -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# 新增函数：仅融合离候选噪声向量最近的 top_k 个图，
# 融合方式沿用原来基于注意力的 ConditionFusion 模块
def fuse_condition_features_for_candidate(candidate_z, batch_graphs, model, train_latents_tensor, top_k,
                                          device="cpu"):
    # candidate_z: shape (1, z_dim)
    # 计算 candidate_z 与训练样本潜变量之间的欧氏距离
    dists = torch.norm(train_latents_tensor - candidate_z, dim=1)  # (num_graphs,)
    # 取最近 top_k 个图（topk 收到最大值，所以对 -dists 取）
    _, indices = torch.topk(-dists, k=top_k)
    selected_graphs = [batch_graphs[idx] for idx in indices]
    # 利用原先的注意力融合方式，仅对选出的图进行融合
    cond_feats_list = []
    for g in selected_graphs:
        if 'skip_multi' not in g.ndata:
            raw = g.ndata['raw_h']
            h1 = model.conv1(g, raw)
            h2 = model.conv2(g, h1)
            h3 = model.conv3(g, h2)
            g.ndata['skip_multi'] = torch.cat([h1, h3], dim=1)
        cond_feat = torch.cat([g.ndata['raw_h'], g.ndata['skip_multi']], dim=1)  # (N, feature_dim)
        cond_feats_list.append(cond_feat)
    # 将选中图的条件特征堆叠成 tensor，形状为 (top_k, N, feature_dim)
    cond_feats_tensor = torch.stack(cond_feats_list, dim=0).to(device)
    fusion_module = ConditionFusion(cond_feats_tensor.shape[-1], hidden_dim=64).to(device)
    # 利用注意力融合，注意力在图维度上归一化后加权求和，
    # 得到形状为 (N, feature_dim) 的个性化条件特征
    fused_cond_feat = fusion_module(cond_feats_tensor)
    return fused_cond_feat


def train_specific_batch(batch_name, config):
    setup_logging(batch_name)
    dataset = MoleculeDataset(config["node_dir"], config["edge_dir"])
    model_output = config["model_output"]
    os.makedirs(model_output, exist_ok=True)
    batches = dataset.split_batches()
    if batch_name not in batches:
        logging.error(f"Batch name '{batch_name}' does not exist! Available batches: {list(batches.keys())}")
        return
    batch_graphs = batches[batch_name]
    in_feats = batch_graphs[0].ndata['raw_h'].shape[1]
    node_count = batch_graphs[0].num_nodes()
    device = torch.device("cpu")
    model = GraphVAE(in_feats, config["hidden_dim"], config["z_dim"], node_count).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    model.train()
    loss_list = []
    mse_loss_list = []
    kl_loss_list = []
    planarity_loss_list = []
    lenandangel_loss_list = []

    num_graphs = len(batch_graphs)
    lambda_planarity = config.get("lambda_planarity", 1.0)
    lambda_lenandangel = config.get("lambda_lenandangel", 1.0)

    for epoch in range(config["num_epochs"]):
        optimizer.zero_grad()
        batched_graph = dgl.batch(batch_graphs).to(device)
        recon_all, mu_all, logvar_all = model.forward_batch(batched_graph)
        node_counts = batched_graph.batch_num_nodes()
        n = node_counts[0]
        gt_list = torch.split(batched_graph.ndata['raw_h'], n)
        recon_list = torch.split(recon_all, n)

        original_loss_total = 0.0
        mse_loss_total = 0.0
        kl_loss_total = 0.0
        plan_loss_total = 0.0
        lenandangel_loss_total = 0.0

        for i in range(len(gt_list)):
            loss, mse_loss_val, kl_loss_val = vae_loss(recon_list[i], gt_list[i],
                                                       mu_all[i:i + 1], logvar_all[i:i + 1],
                                                       beta=config["beta"])
            original_loss_total += loss
            mse_loss_total += mse_loss_val
            kl_loss_total += kl_loss_val

            # 提取本图重构输出中归一化的坐标部分（列1~3，注意第0列为原子序数）
            recon_coords = recon_list[i][:, 1:4]


            ### HIGHLIGHT: 添加重构坐标的统计日志
            rec_min = recon_coords.min().item()
            rec_max = recon_coords.max().item()
            rec_mean = recon_coords.mean().item()
            logging.info(
                f"Epoch {epoch}, Graph {i}, recon_coords stats: min={rec_min:.4f}, max={rec_max:.4f}, mean={rec_mean:.4f}")


            # 芳香环平面性 loss
            plan_loss = compute_aromatic_planarity_loss(batch_graphs[i], recon_coords, config["edge_dir"])
            plan_loss_total += plan_loss

            # 将当前图所有 loss 加入总 loss
            original_loss_total += lambda_planarity * plan_loss

            bond_indices = extract_bond_indices(config, config["edge_dir"])
            angle_indices = extract_angle_indices(config, config["edge_dir"])
            nonbonded_pairs = extract_nonbonded_pairs(config, recon_coords, config["edge_dir"])

            # 键长键角非键原子距离loss
            if bond_indices is not None and bond_indices.numel() > 0:
                bond_loss = compute_modified_bond_length_loss(batch_graphs[i], recon_coords, bond_indices)
            else:
                bond_loss = torch.tensor(0.0, device=recon_coords.device)
            if angle_indices is not None:
                angle_loss = compute_modified_bond_angle_loss(batch_graphs[i], recon_coords, angle_indices)
            else:
                angle_loss = torch.tensor(0.0, device=recon_coords.device)
            if nonbonded_pairs is not None and nonbonded_pairs.numel() > 0:
                nonbond_loss = compute_modified_nonbond_loss(batch_graphs[i], recon_coords, nonbonded_pairs)
            else:
                nonbond_loss = torch.tensor(0.0, device=recon_coords.device)


            lenandangel_loss = bond_loss + angle_loss + nonbond_loss
            lenandangel_loss_total += lenandangel_loss

            original_loss_total += lambda_lenandangel * lenandangel_loss

        num_samples = len(gt_list)
        original_loss_avg = original_loss_total / num_samples
        mse_loss_avg = mse_loss_total / num_samples
        kl_loss_avg = kl_loss_total / num_samples
        plan_loss_avg = plan_loss_total / num_samples
        lenandangel_loss_avg = lenandangel_loss_total / num_samples

        combined_loss = original_loss_avg
        combined_loss.backward()
        optimizer.step()

        loss_list.append(combined_loss.item())
        mse_loss_list.append(mse_loss_avg.item())
        kl_loss_list.append(kl_loss_avg.item())
        planarity_loss_list.append(plan_loss_avg.item())
        lenandangel_loss_list.append(lenandangel_loss_avg.item())

        logging.info(f"Batch: {batch_name}, Epoch: {epoch}, Total Loss: {combined_loss.item():.4f}, "
                     f"MSE Loss: {mse_loss_avg.item():.4f}, KL Loss: {kl_loss_avg.item():.4f}, "
                     f"Planarity Loss: {plan_loss_avg.item():.4f}, Lenandangel Loss: {lenandangel_loss_avg.item():.4f}")
    np.save(os.path.join(model_output, f"{batch_name}_loss.npy"), np.array(loss_list))
    np.save(os.path.join(model_output, f"{batch_name}_mse_loss.npy"), np.array(mse_loss_list))
    np.save(os.path.join(model_output, f"{batch_name}_kl_loss.npy"), np.array(kl_loss_list))
    np.save(os.path.join(model_output, f"{batch_name}_planarity_loss.npy"), np.array(planarity_loss_list))
    np.save(os.path.join(model_output, f"{batch_name}_lenandangel_loss.npy"), np.array(lenandangel_loss_list))
    logging.info(f"Saved training loss curves to {model_output}")
    torch.save(model.state_dict(), os.path.join(config["model_output"], "trained_model_state.pt"))

    # 收集每个图的潜变量（仅取 mu)
    latents = []
    with torch.no_grad():
        for g in batch_graphs:
            g = g.to(device)
            mu, _ = model.encode(g)
            latents.append(mu.cpu().numpy().squeeze(0))
    latents = np.array(latents)

    results_path = os.path.join(model_output, f"{batch_name}_results.npy")
    np.save(results_path, latents)
    logging.info(f"Saved training results to {results_path}")

    # 在潜空间中采样候选向量，并筛选出与训练样本距离较远的向量
    candidate_count = config["candidate_count"]
    noise_threshold = config["noise_threshold"]
    candidate_latents = torch.randn(candidate_count, config["z_dim"])
    train_latents_tensor = torch.tensor(latents, dtype=torch.float32).to(device)
    dists = torch.cdist(candidate_latents, train_latents_tensor, p=2)
    min_dists, _ = torch.min(dists, dim=1)
    mask = min_dists > noise_threshold
    noise_candidates = candidate_latents[mask]
    logging.info(f"Number of noise candidates: {noise_candidates.shape[0]}")

    # 生成新构象时，采用仅融合最近图的条件特征，且融合方式沿用原来的注意力机制
    new_conformers = []
    with torch.no_grad():
        for i in range(noise_candidates.shape[0]):
            global_latent = noise_candidates[i:i + 1]
            latent_flat = model.fc_decode(global_latent)
            latent_node = latent_flat.view(node_count, -1)
            # 计算个性化的条件特征，利用 candidate 噪声对应的 top_k 最近图进行注意力融合
            candidate_fused_cond = fuse_condition_features_for_candidate(global_latent, batch_graphs, model,
                                                                         train_latents_tensor, top_k=config["top_k"],
                                                                         device=device)
            decoder_input = torch.cat([latent_node, candidate_fused_cond], dim=1)
            template_g = batch_graphs[0]
            x = model.gconv1(template_g, decoder_input)
            x = model.gconv2(template_g, x)
            new_feat = model.fc_out(x)
            # 修正生成结果的第一列为模板图中的原子序数
            atomic_numbers = template_g.ndata['raw_h'][:, 0:1]
            new_feat[:, 0:1] = atomic_numbers
            new_feat[:, 1:] = new_feat[:, 1:] * dataset.global_std + dataset.global_mean
            new_conformers.append(new_feat.cpu().numpy())
    new_conformers = np.array(new_conformers)
    logging.info(f"Generated new conformers shape: {new_conformers.shape}")

    new_conformer_output = config["new_conformer_output"]
    os.makedirs(new_conformer_output, exist_ok=True)
    new_conformers_path = os.path.join(new_conformer_output, f"{batch_name}_new_conformers.npy")
    np.save(new_conformers_path, new_conformers)
    logging.info(f"Saved new conformers to {new_conformers_path}")
    del model
    print(f"Finished training batch: {batch_name}\n")


if __name__ == '__main__':
    batch_name = config["batch_name"]
    node_dir = config["node_dir"]
    edge_dir = config["edge_dir"]
    node_output = config["node_output"]
    model_output = config["model_output"]
    new_conformer_output = config["new_conformer_output"]
    dataset = MoleculeDataset(config["node_dir"], config["edge_dir"])
    dataset.save_node_features()
    train_specific_batch(config["batch_name"], config)



