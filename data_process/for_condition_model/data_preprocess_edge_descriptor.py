import os
import numpy as np
from rdkit import Chem

def process_bond_properties(pdb_file, output_dir):
    # 使用 RDKit 读取 pdb 文件
    mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
    if mol is None:
        print("无法读取或解析 PDB 文件，请检查文件路径和文件格式。")
        return

    conformer = mol.GetConformer()
    # 以 pdb 文件名（去除扩展名）作为文件夹名
    folder_name = os.path.splitext(os.path.basename(pdb_file))[0]

    # 1. 计算键类型，并进行 one-hot 编码
    mapping = {"SINGLE": [1, 0, 0, 0],
               "DOUBLE": [0, 1, 0, 0],
               "TRIPLE": [0, 0, 1, 0],
               "OTHER":  [0, 0, 0, 1]}
    bond_types = []
    for bond in mol.GetBonds():
        bt = bond.GetBondType()
        if bt == Chem.BondType.SINGLE:
            bond_type_str = "SINGLE"
        elif bt == Chem.BondType.DOUBLE:
            bond_type_str = "DOUBLE"
        elif bt == Chem.BondType.TRIPLE:
            bond_type_str = "TRIPLE"
        else:
            bond_type_str = "OTHER"
        bond_types.append(mapping[bond_type_str])

    # 2. 计算键芳香性：1 表示芳香，0 表示非芳香
    bond_aromaticity = [1 if bond.GetIsAromatic() else 0 for bond in mol.GetBonds()]

    # 3. 计算键共轭性：1 表示共轭，0 表示非共轭
    bond_conjugation = [1 if bond.GetIsConjugated() else 0 for bond in mol.GetBonds()]

    # 4. 计算键方向：归一化方向向量（包括正向和反向）
    bond_directions = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        pos_i = np.array(conformer.GetAtomPosition(i))
        pos_j = np.array(conformer.GetAtomPosition(j))
        vec = pos_j - pos_i
        norm = np.linalg.norm(vec)
        if norm > 0:
            norm_vec = vec / norm
        else:
            norm_vec = vec
        # 保存正向和反向
        bond_directions.append(norm_vec)
        bond_directions.append(-norm_vec)

    # 5. 计算键是否在环内，以及环的大小
    bond_in_ring = []
    ring_info = mol.GetRingInfo()
    for bond in mol.GetBonds():
        is_in_ring = bond.IsInRing()
        ring_size = 0
        if is_in_ring:
            for ring in ring_info.BondRings():
                if bond.GetIdx() in ring:
                    ring_size = len(ring)
                    break
        bond_in_ring.append((1 if is_in_ring else 0, ring_size))

    # 输出文件路径
    bond_type_output_file = os.path.join(output_dir, f"{folder_name}_bond_types.npy")
    bond_aromaticity_output_file = os.path.join(output_dir, f"{folder_name}_bond_aromaticity.npy")
    bond_conjugation_output_file = os.path.join(output_dir, f"{folder_name}_bond_conjugation.npy")
    bond_directions_output_file = os.path.join(output_dir, f"{folder_name}_bond_directions.npy")
    bond_in_ring_output_file = os.path.join(output_dir, f"{folder_name}_bond_in_ring.npy")

    # 保存为 NumPy 数组文件
    np.save(bond_type_output_file, np.array(bond_types))
    np.save(bond_aromaticity_output_file, np.array(bond_aromaticity))
    np.save(bond_conjugation_output_file, np.array(bond_conjugation))
    np.save(bond_directions_output_file, np.array(bond_directions))
    np.save(bond_in_ring_output_file, np.array(bond_in_ring))

    print("数据保存完毕：")
    print(bond_type_output_file)
    print(bond_aromaticity_output_file)
    print(bond_conjugation_output_file)
    print(bond_directions_output_file)
    print(bond_in_ring_output_file)


    # 输出所有键的信息：键索引及连接的两个原子索引
    print("\n所有键的信息：")
    for bond in mol.GetBonds():
        print(f"键索引: {bond.GetIdx()}, 原子索引: {bond.GetBeginAtomIdx()} - {bond.GetEndAtomIdx()}")


if __name__ == "__main__":
    # 修改 pdb 文件名为实际存在的文件
    pdb_file = r""
    output_dir = r""
    process_bond_properties(pdb_file, output_dir)











