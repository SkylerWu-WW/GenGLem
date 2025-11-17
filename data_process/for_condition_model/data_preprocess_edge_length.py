import os
import numpy as np


def read_coordinates_from_xyz(file_path):
    """
    从 XYZ 文件中读取原子坐标
    :param file_path: XYZ 文件路径
    :return: 原子坐标的 numpy 数组，形状为 (n, 3)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # 第一行是原子数目，第二行是注释，从第三行开始为原子数据
    try:
        atom_count = int(lines[0].strip())
    except:
        print(f"无法解析原子数目: {file_path}")
        return None
    atom_data = lines[2:]
    coordinates = []
    for line in atom_data:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        coordinates.append([x, y, z])
    coordinates = np.array(coordinates)
    if len(coordinates) != atom_count:
        print(f"警告：文件 {file_path} 实际读取的原子数与文件声明不一致！")
    return coordinates


def calculate_bond_lengths(coordinates, bonds):
    """
    根据提供的原子坐标和键的原子索引计算每个键的长度
    :param coordinates: numpy 数组，形状为 (n, 3)
    :param bonds: 键列表，每个元素为一个元组 (i, j)
    :return: 键长列表，每个元素为 (i, j, 长度)
    """
    bond_lengths = []
    for i, j in bonds:
        # 检查索引是否超出范围
        if i >= len(coordinates) or j >= len(coordinates):
            print(f"索引越界: {i} 或 {j} (坐标数目: {len(coordinates)})")
            continue
        length = np.linalg.norm(coordinates[i] - coordinates[j])
        bond_lengths.append((i, j, length))
    return bond_lengths


def process_all_xyz(input_dir, output_dir, bonds):
    """
    遍历 input_dir 下所有 XYZ 文件，计算键长并保存为 npy 文件
    :param input_dir: 存放 XYZ 文件的文件夹路径
    :param output_dir: 计算结果输出文件夹路径
    :param bonds: 键列表，每个键用 (i, j) 表示两个原子索引
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xyz_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.xyz')]
    print(f"找到 {len(xyz_files)} 个 xyz 文件。")

    for xyz_file in xyz_files:
        file_path = os.path.join(input_dir, xyz_file)
        coordinates = read_coordinates_from_xyz(file_path)
        if coordinates is None:
            continue

        bond_lengths = calculate_bond_lengths(coordinates, bonds)
        # 输出文件名：原始文件名去掉扩展名，加上 _bond_lengths.npy
        base_name = os.path.splitext(xyz_file)[0]
        output_file = os.path.join(output_dir, base_name + "_bond_lengths.npy")
        np.save(output_file, np.array(bond_lengths))

        print(f"文件 {xyz_file} 处理完毕：保存键长至 {output_file}")


if __name__ == "__main__":
    # 文件夹路径请根据实际情况修改
    input_folder = r"C:\Users\M340024\Graph_for_conformer\condition_data\dataset_costruc"
    output_folder = r"C:\Users\M340024\Graph_for_conformer\condition_data\npy_edge_dataset_sum"

    # 预先定义好键对应的原子索引（注意：原子索引从 0 开始）
    # 例如：(0, 1) 表示第 0 个原子和第 1 个原子之间存在一条键
    bonds = [
        (7, 6),
        (7, 8),
        (7, 9),
        (6, 4),
        (6, 10),
        (29, 27),
        (27, 26),
        (27, 28),
        (33, 32),
        (33, 34),
        (33, 35),
        (32, 10),
        (32, 30),
        (4, 3),
        (4, 5),
        (4, 24),
        (30, 18),
        (30, 26),
        (30, 31),
        (26, 24),
        (11, 10),
        (12, 3),
        (12, 10),
        (12, 13),
        (12, 14),
        (2, 1),
        (18, 14),
        (18, 19),
        (18, 20),
        (20, 21),
        (20, 24),
        (21, 22),
        (21, 23),
        (25, 24),
        (0, 1),
        (3, 1),
        (14, 15),
        (15, 16),
        (15, 17)
    ]


    process_all_xyz(input_folder, output_folder, bonds)
