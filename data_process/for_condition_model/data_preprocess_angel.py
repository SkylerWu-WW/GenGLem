import os
import numpy as np


class AngleCalculator:
    def __init__(self, input_dir, output_dir):
        """
        初始化类，设置输入（XYZ 文件所在目录）和输出目录
        :param input_dir: 存放XYZ文件的路径
        :param output_dir: 输出结果保存路径
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def read_coordinates(self, file_path):
        """
        从 XYZ 文件中读取原子坐标，返回一个 numpy 数组（索引从 0 开始）
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()
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
            print(f"警告：文件 {file_path} 实际读取的原子数与声明的不一致！")
        return coordinates

    def calculate_manual_bond_angle(self, coordinates, manual_angle_list):
        """
        根据手动输入的角原子索引三元组计算角度。
        :param coordinates: 原子坐标的 numpy 数组（索引从 0 开始）
        :param manual_angle_list: 列表，每个元素为一个三元组 (i, j, k)
                                  表示以 j 为顶点，由 i-j-k 构成的角
        :return: 角度列表，每个元素为 (i, j, k, angle)，单位为弧度
        """
        angle_list = []
        for i, j, k in manual_angle_list:
            vec1 = coordinates[i] - coordinates[j]
            vec2 = coordinates[k] - coordinates[j]
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                angle = None
                print(f"警告：原子 {i} 或原子 {k} 与顶点 {j} 距离为 0")
            else:
                cos_val = np.dot(vec1, vec2) / (norm1 * norm2)
                cos_val = np.clip(cos_val, -1.0, 1.0)
                angle = np.arccos(cos_val)
            angle_list.append((i, j, k, angle))
        return angle_list

    def process_xyz_files(self):
        """
        遍历输入目录下的所有 XYZ 文件，计算每个文件中手动指定角度，并将结果保存为 .npy 文件
        """
        xyz_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.xyz')]
        print(f"找到 {len(xyz_files)} 个 XYZ 文件。")

        # 手动定义角度计算的原子索引三元组（注意：这里的索引均为 0 开始）
        # 例如：(2, 5, 10) 表示以第 5 个原子为顶点，由原子 2-5-10 构成的角
        manual_angle_list = [
            (3, 4, 5),
            (3, 12, 13),
            (4, 24, 25),
            (5, 4, 6),
            (5, 4, 24),
            (6, 10, 11),
            (10, 12, 13),
            (11, 10, 12),
            (11, 10, 32),
            (13, 12, 14),
            (14, 18, 19),
            (18, 30, 31),
            (19, 18, 20),
            (19, 18, 30),
            (20, 24, 25),
            (25, 24, 26),
            (26, 30, 31),
            (31, 30, 32),
            (0, 1, 2),
            (0, 1, 3),
            (1, 3, 4),
            (1, 3, 12),
            (2, 1, 3),
            (3, 4, 6),
            (3, 4, 24),
            (3, 12, 10),
            (3, 12, 14),
            (4, 3, 12),
            (4, 6, 7),
            (4, 6, 10),
            (4, 24, 20),
            (4, 24, 26),
            (6, 4, 24),
            (6, 7, 8),
            (6, 7, 9),
            (6, 10, 12),
            (6, 10, 32),
            (7, 6, 10),
            (8, 7, 9),
            (10, 12, 14),
            (10, 32, 30),
            (10, 32, 33),
            (12, 10, 32),
            (12, 14, 15),
            (12, 14, 18),
            (14, 15, 16),
            (14, 15, 17),
            (14, 18, 20),
            (14, 18, 30),
            (15, 14, 18),
            (16, 15, 17),
            (18, 20, 21),
            (18, 20, 24),
            (18, 30, 26),
            (18, 30, 32),
            (20, 18, 30),
            (20, 21, 22),
            (20, 21, 23),
            (20, 24, 26),
            (21, 20, 24),
            (22, 21, 23),
            (24, 26, 27),
            (24, 26, 30),
            (26, 27, 28),
            (26, 27, 29),
            (26, 30, 32),
            (27, 26, 30),
            (28, 27, 29),
            (30, 32, 33),
            (32, 33, 34),
            (32, 33, 35),
            (34, 33, 35)
        ]

        for xyz_file in xyz_files:
            file_path = os.path.join(self.input_dir, xyz_file)
            coordinates = self.read_coordinates(file_path)
            if coordinates is None:
                continue

            angles = self.calculate_manual_bond_angle(coordinates, manual_angle_list)
            output_file = os.path.join(self.output_dir, os.path.splitext(xyz_file)[0] + "_angles.npy")
            np.save(output_file, np.array(angles))
            print(f"文件 {xyz_file} 处理完毕，角度信息保存在：{output_file}")


if __name__ == "__main__":
    input_folder = r""
    output_folder = r""
    calc = AngleCalculator(input_folder, output_folder)
    calc.process_xyz_files()


