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

    def calculate_twosided_angle(self, folder_name, mol, coordinates):
        """
        计算每组四个原子之间的二面角。
        :param folder_name: 分子文件夹名称。
        :param mol: RDKit 分子对象。
        :param coordinates: 原子坐标数组。
        :return: 每组四个原子之间的二面角列表。
        """
        bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
        twosided_angles = []
        for bond in mol.GetBonds():
            node1, node2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            for neighbor1 in mol.GetAtomWithIdx(node1).GetNeighbors():
                if neighbor1.GetIdx() == node2:
                    continue
                for neighbor2 in mol.GetAtomWithIdx(node2).GetNeighbors():
                    if neighbor2.GetIdx() == node1:
                        continue
                    node3, node4 = neighbor1.GetIdx(), neighbor2.GetIdx()
                    vec1 = coordinates[node1] - coordinates[node3]
                    vec2 = coordinates[node2] - coordinates[node1]
                    vec3 = coordinates[node4] - coordinates[node2]
                    normal1 = np.cross(vec1, vec2)
                    normal2 = np.cross(vec2, vec3)
                    angle = np.arccos(np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2)))
                    twosided_angles.append((node3, node1, node2, node4, angle))
        return twosided_angles

    def process_molecules(self):
        """
        处理所有分子文件夹，并计算每个分子的键长度、键类型、键的芳香性、键的方向、键的共轭性、键的环信息和键的夹角。
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

                bond_angles = self.calculate_bond_angle(folder_name, mol, coordinates)
                twosided_angles = self.calculate_twosided_angle(folder_name, mol, coordinates)

                # 保存键的夹角信息为 NumPy 数组
                bond_angle_output_file = os.path.join(self.output_dir, f"{folder_name}_{xyz_file}_bond_angles.npy")
                np.save(bond_angle_output_file, np.array(bond_angles))

                # 保存键的二面角信息为 NumPy 数组
                twosided_angle_output_file = os.path.join(self.output_dir, f"{folder_name}_{xyz_file}_twosided_angles.npy")
                np.save(twosided_angle_output_file, np.array(twosided_angles))


# 示例用法
base_dir = r""
excel_file_path = r""
output_dir = r""

bond_data = get_bond_data(base_dir, excel_file_path, output_dir)
bond_data.process_molecules()











