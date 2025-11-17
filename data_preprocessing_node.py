import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import pandas as pd
import re



# 元素符号到原子序数的映射字典
element_to_atomic_number = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
    'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}

class XYZReader:
    def __init__(self, folder_path):
        """
        初始化 XYZReader 类。
        :param folder_path: 包含 .xyz 文件的文件夹路径。
        """
        self.folder_path = folder_path
        self.atom_count = None
        self.atom_numbers = []
        self.coordinates = []

    def read_xyz_files(self):
        """
        读取文件夹中的所有 .xyz 文件，并提取原子坐标和原子序数。
        """
        # 获取文件夹中的所有 .xyz 文件
        xyz_files = [f for f in os.listdir(self.folder_path) if f.endswith('.xyz')]

        for file_name in xyz_files:
            file_path = os.path.join(self.folder_path, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()

                # 第一行是原子数量
                self.atom_count = int(lines[0].strip())

                # 跳过第二行（空白行）
                atom_data = lines[2:]

                # 初始化原子序数和坐标列表
                atom_numbers = []
                coords = []

                for line in atom_data:
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue  # 跳过格式不正确的行
                    element_symbol = parts[0]
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    atom_number = element_to_atomic_number.get(element_symbol, None)
                    if atom_number is None:
                        raise ValueError(f"Unknown element symbol: {element_symbol}")
                    atom_numbers.append(atom_number)
                    coords.append([x, y, z])

                # 保存原子序数和坐标
                self.atom_numbers.append(atom_numbers)
                self.coordinates.append(coords)

        # 将原子序数和坐标转换为 NumPy 数组
        self.atom_numbers = np.array(self.atom_numbers)
        self.coordinates = np.array(self.coordinates)


    def get_data(self):
        """
        返回提取的数据。
        :return: 原子序数和坐标。
        """
        return self.atom_numbers, self.coordinates

def process_mol(base_dir, folder_name, output_dir):
    """
    处理指定的分子文件夹。
    :param base_dir: 基目录路径。
    :param folder_name: 分子文件夹名称。
    :param output_dir: 输出目录路径。
    """
    folder_path = os.path.join(base_dir, folder_name)
    # 获取所有 .xyz 文件，并使用 lambda 从文件名中提取数字排序
    xyz_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.xyz')],
                       key=lambda f: int(re.search(r'frame_(\d+)', f).group(1)) if re.search(r'frame_(\d+)', f) else -1)
    xyz_reader = XYZReader(folder_path)
    xyz_reader.read_xyz_files()
    atom_numbers, coordinates = xyz_reader.get_data()
    print(f"Atom numbers for {folder_name}: {atom_numbers}")
    print(f"Coordinates for {folder_name}: {coordinates}")

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 保存数据到指定目录
    for file_name, (anum, coord) in zip(xyz_files, zip(atom_numbers, coordinates)):
        frame_tag = file_name.replace('.xyz', '')
        np.save(os.path.join(output_dir, f"{folder_name}_{frame_tag}_atom_numbers.npy"), np.array(anum))
        np.save(os.path.join(output_dir, f"{folder_name}_{frame_tag}_coordinates.npy"), np.array(coord))


class SMILESProcessor:
    def __init__(self, excel_file_path):
        """
        初始化 SMILESProcessor 类。
        :param excel_file_path: Excel 文件的路径。
        """
        self.excel_file_path = excel_file_path
        self.data = None

    def load_data(self):
        """
        从 Excel 文件中加载数据。
        """
        if not os.path.exists(self.excel_file_path):
            raise FileNotFoundError(f"File not found: {self.excel_file_path}")

        self.data = pd.read_excel(self.excel_file_path, header=0)
        if self.data is None:
            raise ValueError("Failed to load data from Excel file")

    def get_smiles(self):
        """
        获取 SMILES 数据。
        :return: 序号和 SMILES 的列表。
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        # 假设第一列是序号，第三列是 SMILES
        return list(zip(self.data.iloc[:, 0], self.data.iloc[:, 2]))

    @staticmethod
    def calculate_formal_charges(smiles):
        """
        使用 RDKit 计算分子的形式电荷。
        :param smiles: 分子的 SMILES 表示。
        :return: 形式电荷列表。
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        # 添加氢原子
        mol = Chem.AddHs(mol)

        # 计算形式电荷
        formal_charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
        return formal_charges

    def process_and_save_formal_charges(self, output_dir):
        """
        处理 SMILES 数据并保存形式电荷。
        :param output_dir: 输出目录路径。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        smiles_data = self.get_smiles()
        for idx, smiles in smiles_data:
            try:
                formal_charges = self.calculate_formal_charges(smiles)
                np.save(os.path.join(output_dir, f"No.{idx}_formalcharge.npy"), np.array(formal_charges))
            except ValueError as e:
                print(f"No. {idx}: SMILES = {smiles}, Error = {e}")


    @staticmethod
    def calculate_hybridization(smiles):
        """
        使用 RDKit 的子结构匹配功能计算分子中每个原子的杂化状态。
        :param smiles: 分子的 SMILES 表示。
        :return: 杂化状态列表。
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        mol = Chem.AddHs(mol)

        # 定义杂化状态的 SMARTS 查询
        hybridization_queries = {
            'S': '[^0]',
            'SP': '[^1]',
            'SP2': '[^2]',
            'SP3': '[^3]',
            'SP3D': '[^4]',
            'SP3D2': '[^5]'
        }

        # 初始化杂化状态列表
        hybridizations = [None] * mol.GetNumAtoms()

        # 遍历每个杂化状态查询
        for hybridization, query in hybridization_queries.items():
            query_mol = Chem.MolFromSmarts(query)
            matches = mol.GetSubstructMatches(query_mol)
            for match in matches:
                hybridizations[match[0]] = hybridization

        return hybridizations

    @staticmethod
    def encode_hybridization(hybridizations):
        """
        编码杂化状态。
        :param hybridizations: 杂化状态列表。
        :return: 编码后的杂化状态。
        """
        # 独热编码
        hybridization_dict = {
            'NONE': [1, 0, 0, 0, 0, 0, 0],
            'S': [0, 1, 0, 0, 0, 0, 0],
            'SP': [0, 0, 1, 0, 0, 0, 0],
            'SP2': [0, 0, 0, 1, 0, 0, 0],
            'SP3': [0, 0, 0, 0, 1, 0, 0],
            'SP3D': [0, 0, 0, 0, 0, 1, 0],
            'SP3D2': [0, 0, 0, 0, 0, 0, 1]
        }
        encoded_hybridizations = [hybridization_dict.get(h, hybridization_dict['NONE']) for h in hybridizations]

        return encoded_hybridizations

    def process_and_save_hybridizations(self, output_dir):
        """
        处理 SMILES 数据并保存杂化状态。
        :param output_dir: 输出目录路径。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        smiles_data = self.get_smiles()
        for idx, smiles in smiles_data:
            try:
                hybridizations = self.calculate_hybridization(smiles)
                encoded_hybridizations = self.encode_hybridization(hybridizations)
                np.save(os.path.join(output_dir, f"No.{idx}_hybridizations.npy"), np.array(encoded_hybridizations))
            except ValueError as e:
                print(f"No. {idx}: SMILES = {smiles}, Error = {e}")


    @staticmethod
    def calculate_is_aromatic(smiles):
        """
        使用 RDKit 判断分子中每个原子是否处于芳香环中。
        :param smiles: 分子的 SMILES 表示。
        :return: 每个原子是否处于芳香环的布尔列表。
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        # 添加氢原子
        mol = Chem.AddHs(mol)

        # 计算每个原子是否处于芳香环中
        is_aromatic = np.array([atom.GetIsAromatic() for atom in mol.GetAtoms()], dtype=int)
        return is_aromatic

    def process_and_save_is_aromatic(self, output_dir):
        """
        处理 SMILES 数据并保存每个原子是否处于芳香环中的信息。
        :param output_dir: 输出目录路径。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        smiles_data = self.get_smiles()
        for idx, smiles in smiles_data:
            try:
                is_aromatic = self.calculate_is_aromatic(smiles)
                np.save(os.path.join(output_dir, f"No.{idx}_is_aromatic.npy"), np.array(is_aromatic))
            except ValueError as e:
                print(f"No. {idx}: SMILES = {smiles}, Error = {e}")


    @staticmethod
    def calculate_valence_electrons(smiles):
        """
        使用 RDKit 计算分子中每个原子的 VSEPR 特征。
        :param smiles: 分子的 SMILES 表示。
        :return: VSEPR 特征列表。
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        # 添加氢原子
        mol = Chem.AddHs(mol)

        # 计算每个原子的 VSEPR 特征
        for atom in mol.GetAtoms():
            # 原子的价电子数
            valence_electrons = [Chem.GetPeriodicTable().GetNOuterElecs(atom.GetAtomicNum()) for atom in mol.GetAtoms()]

        return valence_electrons

    def process_and_save_valence_electrons(self, output_dir):
        """
        处理 SMILES 数据并保存每个原子的 VSEPR 特征。
        :param output_dir: 输出目录路径。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        smiles_data = self.get_smiles()
        for idx, smiles in smiles_data:
            try:
                valence_electrons = self.calculate_valence_electrons(smiles)
                np.save(os.path.join(output_dir, f"No.{idx}_valence_electrons.npy"), np.array(valence_electrons))
            except ValueError as e:
                print(f"No. {idx}: SMILES = {smiles}, Error = {e}")


    @staticmethod
    def calculate_explicit_valence(smiles):
        """
        使用 RDKit 计算分子中每个原子的 VSEPR 特征。
        :param smiles: 分子的 SMILES 表示。
        :return: VSEPR 特征列表。
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        # 添加氢原子
        mol = Chem.AddHs(mol)

        # 计算每个原子的 VSEPR 特征
        for atom in mol.GetAtoms():
            # 显式价电子数
            explicit_valence = [atom.GetExplicitValence() for atom in mol.GetAtoms()]
        return explicit_valence

    def process_and_save_explicit_valence(self, output_dir):
        """
        处理 SMILES 数据并保存每个原子的 VSEPR 特征。
        :param output_dir: 输出目录路径。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        smiles_data = self.get_smiles()
        for idx, smiles in smiles_data:
            try:
                explicit_valence = self.calculate_explicit_valence(smiles)
                np.save(os.path.join(output_dir, f"No.{idx}_explicit_valence.npy"), np.array(explicit_valence))
            except ValueError as e:
                print(f"No. {idx}: SMILES = {smiles}, Error = {e}")


    @staticmethod
    def calculate_lone_pairs(smiles):
        """
        使用 RDKit 计算分子中每个原子的 VSEPR 特征。
        :param smiles: 分子的 SMILES 表示。
        :return: VSEPR 特征列表。
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        # 添加氢原子
        mol = Chem.AddHs(mol)

        # 计算每个原子的孤对电子数
        lone_pairs = []
        for atom in mol.GetAtoms():
            # 原子的价电子数
            valence_electrons = Chem.GetPeriodicTable().GetNOuterElecs(atom.GetAtomicNum())
            # 显式价电子数
            explicit_valence = atom.GetExplicitValence()
            # 孤对电子数
            lone_pair = (valence_electrons - explicit_valence) // 2
            lone_pairs.append(lone_pair)

        return lone_pairs

    def process_and_save_lone_pairs(self, output_dir):
        """
        处理 SMILES 数据并保存每个原子的 VSEPR 特征。
        :param output_dir: 输出目录路径。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        smiles_data = self.get_smiles()
        for idx, smiles in smiles_data:
            try:
                lone_pairs = self.calculate_lone_pairs(smiles)
                np.save(os.path.join(output_dir, f"No.{idx}_lone_pairs.npy"), np.array(lone_pairs))
            except ValueError as e:
                print(f"No. {idx}: SMILES = {smiles}, Error = {e}")




if __name__ == "__main__":
    base_dir = r"C:\Users\M340024\Graph_for_conformer\lammps_data"
    output_dir = r"C:\Users\M340024\Graph_for_conformer\dataset_for_Graph_node"
    excel_file_path = r"C:\Users\M340024\Graph_for_conformer\SMILES.xlsx"

    # 创建 SMILESProcessor 对象并加载数据
    smiles_processor = SMILESProcessor(excel_file_path)
    smiles_processor.load_data()

    # 处理 SMILES 数据并保存形式电荷
    smiles_processor.process_and_save_formal_charges(output_dir)

    # 处理 SMILES 数据并保存杂化状态
    smiles_processor.process_and_save_hybridizations(output_dir)

    # 处理 SMILES 数据并保存每个原子是否处于芳香环中的信息
    smiles_processor.process_and_save_is_aromatic(output_dir)

    # 处理 SMILES 数据并保存每个原子的原子价
    smiles_processor.process_and_save_valence_electrons(output_dir)

    # 处理 SMILES 数据并保存每个原子的显示价电子
    smiles_processor.process_and_save_explicit_valence(output_dir)

    # 处理 SMILES 数据并保存每个原子的孤对电子
    smiles_processor.process_and_save_lone_pairs(output_dir)


    # 获取所有分子文件夹名称
    folder_names = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    # 动态处理每个分子文件夹
    for folder_name in folder_names:
        print(f"Processing folder: {folder_name}")
        process_mol(base_dir, folder_name, output_dir)
        print(f"Finished processing folder: {folder_name}")

























































































