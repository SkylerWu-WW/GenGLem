import os
import numpy as np
from rdkit import Chem
import pandas as pd

class get_bond_data:
    def __init__(self, base_dir, excel_file_path, output_dir):
        """
        初始化类，读取 Excel 文件并创建输出目录。
        :param base_dir: 包含所有分子文件夹的根目录路径。
        :param excel_file_path: 包含分子序号和 SMILES 的 Excel 文件路径。
        :param output_dir: 输出目录路径。
        """
        self.base_dir = base_dir
        self.excel_file_path = excel_file_path
        self.output_dir = output_dir
        self.smiles_data = pd.read_excel(self.excel_file_path, header=0)
        self.smiles_dict = {row[0]: row[2] for index, row in self.smiles_data.iterrows()}
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def calculate_bond_length(self, folder_name, mol, coordinates):
        """
        计算键长并保存为 NumPy 数组。
        :param folder_name: 分子文件夹名称。
        :param mol: RDKit 分子对象。
        :param coordinates: 原子坐标数组。
        """
        bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
        bond_lengths = []
        for bond in bonds:
            node1, node2 = bond
            bond_length = np.linalg.norm(coordinates[node1] - coordinates[node2])
            bond_lengths.append((node1, node2, bond_length))
        return bond_lengths

    def calculate_type_of_bond(self, mol):
        """
        计算键的类型并返回键类型列表。
        :param mol: RDKit 分子对象。
        :return: 键类型列表，每个键类型用字符串表示（'SINGLE', 'DOUBLE', 'TRIPLE'）。
        """
        bond_types = []
        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            if bond_type == Chem.BondType.SINGLE:
                bond_types.append('SINGLE')
            elif bond_type == Chem.BondType.DOUBLE:
                bond_types.append('DOUBLE')
            elif bond_type == Chem.BondType.TRIPLE:
                bond_types.append('TRIPLE')
            else:
                bond_types.append('OTHER')  # 用于其他键类型
        return bond_types

    def bond_in_aromatic(self, mol):
        """
        判断每个键是否在芳香环中，并返回整数列表。
        :param mol: RDKit 分子对象。
        :return: 整数列表，表示每个键是否在芳香环中（0 表示不在，1 表示在）。
        """
        bond_aromaticity = []
        for bond in mol.GetBonds():
            if bond.GetIsAromatic():
                bond_aromaticity.append(1)
            else:
                bond_aromaticity.append(0)
        return bond_aromaticity

    def calculate_bond_direction(self, mol, coordinates):
        """
        计算键的方向向量并返回归一化后的方向向量列表。
        :param mol: RDKit 分子对象。
        :param coordinates: 原子坐标数组。
        :return: 归一化后的方向向量列表。
        """
        bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
        bond_directions = []
        for bond in bonds:
            node1, node2 = bond
            direction_vector = coordinates[node2] - coordinates[node1]
            norm = np.linalg.norm(direction_vector)
            if norm > 0:
                normalized_direction = direction_vector / norm
            else:
                normalized_direction = direction_vector  # 避免除以零
            bond_directions.append(normalized_direction)
            bond_directions.append(-normalized_direction)  # 添加反向边的方向
        return bond_directions

    def bond_conjugation(self, mol):
        """
        判断每个键是否参与共轭，并返回布尔列表。
        :param mol: RDKit 分子对象。
        :return: 布尔列表，表示每个键是否参与共轭。
        """
        bond_conjugation = []
        for bond in mol.GetBonds():
            bond_conjugation.append(int(bond.GetIsConjugated()))  # 转换为整数值
        return bond_conjugation

    def bond_in_ring(self, mol):
        """
        判断每个键是否在环中，并返回环的大小信息。
        :param mol: RDKit 分子对象。
        :return: 列表，每个元素是一个元组 (是否在环中, 环的大小)。
        """
        bond_ring_info = []
        for bond in mol.GetBonds():
            is_in_ring = bond.IsInRing()
            ring_size = 0
            if is_in_ring:
                ring_info = mol.GetRingInfo()
                for ring in ring_info.BondRings():
                    if bond.GetIdx() in ring:
                        ring_size = len(ring)
                        break
            bond_ring_info.append((int(is_in_ring), ring_size))  # 转换为整数值
        return bond_ring_info

    def calculate_bond_angle(self, folder_name, mol, coordinates):
        """
        计算每对相邻键之间的夹角。
        :param folder_name: 分子文件夹名称。
        :param mol: RDKit 分子对象。
        :param coordinates: 原子坐标数组。
        :return: 每对相邻键之间的夹角列表。
        """
        bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
        bond_angles = []
        for bond in mol.GetBonds():
            node1, node2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            for neighbor in mol.GetAtomWithIdx(node2).GetNeighbors():
                if neighbor.GetIdx() == node1:
                    continue
                node3 = neighbor.GetIdx()
                vec1 = coordinates[node2] - coordinates[node1]
                vec2 = coordinates[node3] - coordinates[node2]
                angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
                bond_angles.append((node1, node2, node3, angle))
        return bond_angles

    def process_molecules(self):
        """
        处理所有分子文件夹，并计算每个分子的键长度。
        """
        for idx, smiles in self.smiles_dict.items():
            folder_name = f"lammps_{idx}"
            folder_path = os.path.join(self.base_dir, folder_name)
            if not os.path.exists(folder_path):
                print(f"Folder not found: {folder_path}")
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Invalid SMILES string for index {idx}: {smiles}")
                continue
            mol = Chem.AddHs(mol)

            xyz_files = [f for f in os.listdir(folder_path) if f.endswith('.xyz')]
            for xyz_file in xyz_files:
                file_path = os.path.join(folder_path, xyz_file)
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                atom_count = int(lines[0].strip())
                atom_data = lines[2:]
                coordinates = []
                for line in atom_data:
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    coordinates.append([x, y, z])
                coordinates = np.array(coordinates)

                if len(coordinates) != mol.GetNumAtoms():
                    print(f"Atom count mismatch for {folder_name}/{xyz_file}")
                    continue

                bond_lengths = self.calculate_bond_length(folder_name, mol, coordinates)
                bond_angles = self.calculate_bond_angle(folder_name, mol, coordinates)
                # 保存键长信息为 NumPy 数组
                bond_length_output_file = os.path.join(self.output_dir, f"{folder_name}_{xyz_file}_bond_lengths.npy")
                np.save(bond_length_output_file, np.array(bond_lengths))
                # 保存键的夹角信息为 NumPy 数组
                bond_angle_output_file = os.path.join(self.output_dir, f"{folder_name}_{xyz_file}_bond_angles.npy")
                np.save(bond_angle_output_file, np.array(bond_angles))
            # 在处理完所有 xyz 文件后，保存键类型\键的芳香性信息\方向为 NumPy 数组
            bond_types = self.calculate_type_of_bond(mol)
            bond_aromaticity = self.bond_in_aromatic(mol)
            bond_directions = self.calculate_bond_direction(mol, coordinates)
            bond_conjugation = self.bond_conjugation(mol)
            bond_in_ring = self.bond_in_ring(mol)

            # 对键类型进行独热编码处理
            mapping = {"SINGLE": [1, 0, 0, 0], "DOUBLE": [0, 1, 0, 0], "TRIPLE": [0, 0, 1, 0], "OTHER": [0, 0, 0, 1]}
            onehot_bond_types = [mapping.get(bt, [0, 0, 0, 1]) for bt in bond_types]

            bond_type_output_file = os.path.join(self.output_dir, f"{folder_name}_bond_types.npy")
            bond_aromaticity_output_file = os.path.join(self.output_dir, f"{folder_name}_bond_aromaticity.npy")
            bond_conjugation_output_file = os.path.join(self.output_dir, f"{folder_name}_bond_conjugation.npy")
            bond_directions_output_file = os.path.join(self.output_dir, f"{folder_name}_bond_directions.npy")
            bond_in_ring_output_file = os.path.join(self.output_dir, f"{folder_name}_bond_in_ring.npy")
            np.save(bond_type_output_file, np.array(onehot_bond_types))
            np.save(bond_aromaticity_output_file, np.array(bond_aromaticity))
            np.save(bond_directions_output_file, np.array(bond_directions))
            np.save(bond_conjugation_output_file, np.array(bond_conjugation))
            np.save(bond_in_ring_output_file, np.array(bond_in_ring))


# 示例用法
base_dir = r""
excel_file_path = r""
output_dir = r""

bond_data = get_bond_data(base_dir, excel_file_path, output_dir)
bond_data.process_molecules()

















































