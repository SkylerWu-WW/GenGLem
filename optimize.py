import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D


def update_mol_coordinates(mol, new_coords):
    """
    利用 new_coords 更新 mol 对象的构象坐标。
    new_coords 的形状为 (N, 4)，其中第一列可忽略，后三列表示新的 (x, y, z) 坐标。
    """
    natoms = mol.GetNumAtoms()
    if new_coords.shape[0] != natoms:
        raise ValueError("新的坐标原子数与原 MOL 文件中原子数不匹配！")

    # 创建新的构象并设置原子坐标
    conf = Chem.Conformer(natoms)
    for i in range(natoms):
        x, y, z = new_coords[i, 1:4]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.RemoveAllConformers()
    mol.AddConformer(conf)
    return mol


def optimize_molecule_uff(mol, maxIters=300):
    """
    利用 RDKit 的 UFF 力场对分子进行能量优化，
    返回优化的状态代码（0 表示收敛）和优化后能量值。
    """
    result = AllChem.UFFOptimizeMolecule(mol, maxIters=maxIters)
    forcefield = AllChem.UFFGetMoleculeForceField(mol)
    energy = forcefield.CalcEnergy()
    return result, energy


def optimize_molecule_mmff(mol, maxIters=500):
    """
    利用 RDKit 的 MMFF94 力场对分子进行能量优化，
    返回优化状态代码和优化后能量值。
    """
    # 尝试直接调用 MMFFOptimizeMolecule
    try:
        result = AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94', maxIters=maxIters)
        forcefield = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
        energy = forcefield.CalcEnergy()
        return result, energy
    except Exception as e:
        print(f"MMFFOptimizeMolecule 直接调用失败：{e}")

    # 如果直接调用失败，尝试通过力场对象进行优化
    try:
        properties = AllChem.MMFFGetMoleculeProperties(mol)
        forcefield = AllChem.MMFFGetMoleculeForceField(mol, properties)
        result = forcefield.Minimize(maxIts=maxIters)
        energy = forcefield.CalcEnergy()
        return result, energy
    except Exception as e:
        print(f"通过力场对象优化失败：{e}")
        raise


def create_gaussian_input(mol, output_file):
    """
    创建 Gaussian 输入文件。
    """
    # 获取原子信息和坐标
    atoms = mol.GetAtoms()
    conformer = mol.GetConformer()
    positions = conformer.GetPositions()

    # 写入 Gaussian 输入文件
    with open(output_file, "w") as f:
        f.write("%nprocshared=10\n")
        f.write("#P PBE1PBE/6-31G(d) Opt\n\n")
        f.write("Title Molecule for QSPR Analysis\n\n")
        f.write("0 1\n")
        for atom, pos in zip(atoms, positions):
            symbol = atom.GetSymbol()
            x, y, z = pos
            f.write(f"{symbol:<4}{x:12.6f}{y:12.6f}{z:12.6f}\n")
        f.write("\nEnd of input\n")



def main():
    # 文件和目录路径设置，请根据实际情况修改
    npy_file = r"C:\Users\M340024\Graph_for_conformer\evaluate\evaluate\72\GGVAE\lammps_72_new_conformers.npy"
    mol_file = r"C:\Users\M340024\Graph_for_conformer\Rdkit_coordinates\72.mol"
    output_dir = r"C:\Users\M340024\Graph_for_conformer\evaluate\evaluate\72\GGVAE"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载原始 MOL 文件（该文件中包含完整键信息）
    mol_orig = Chem.MolFromMolFile(mol_file, removeHs=False)
    if mol_orig is None:
        raise ValueError(f"加载 MOL 文件失败：{mol_file}")

    # 加载包含多个构象的坐标文件（numpy 格式），形状为 (num_conformers, N, 4)
    conformers = np.load(npy_file)
    num_conf = conformers.shape[0]
    print(f"加载到 {num_conf} 个构象。")

    all_results = []

    # 遍历每个构象：更新坐标 -> 保存更新后的 MOL 文件 -> MMFF94 能量优化 -> 保存优化后 MOL 文件
    for idx in range(num_conf):
        new_coords = conformers[idx]  # 新构象，形状为 (N, 4)

        # 拷贝原始分子避免直接修改（浅拷贝即可）
        mol_copy = Chem.Mol(mol_orig)
        mol_updated = update_mol_coordinates(mol_copy, new_coords)

        # 保存更新后的 MOL 文件（未优化）
        out_updated = os.path.join(output_dir, f"25_updated_conformer_{idx}.mol")
        Chem.MolToMolFile(mol_updated, out_updated)
        print(f"构象 {idx} 已保存更新后的分子到: {out_updated}")

        method = "MMFF94"
        try:
            opt_result, opt_energy = optimize_molecule_mmff(mol_updated, maxIters=500)
        except Exception as e:
            print(f"构象 {idx} 在 {method} 优化时发生错误：{e}")
            continue

        print(f"构象 {idx} {method} 优化结果代码：{opt_result}, 优化后能量：{opt_energy:.4f}")

        # 保存优化后的 MOL 文件
        out_optimized = os.path.join(output_dir, f"25_optimized_conformer_{idx}.mol")
        Chem.MolToMolFile(mol_updated, out_optimized)
        print(f"构象 {idx} 优化后的分子已保存到: {out_optimized}")

        # 将结果添加到列表中
        all_results.append({
            "conformer_index": idx,
            "optimization_result": opt_result,
            "optimized_energy": opt_energy
        })


        # 创建 Gaussian 输入文件
        gaussian_input_file = os.path.join(output_dir, f"25_optimized_conformer_{idx}.gjf")
        create_gaussian_input(mol_updated, gaussian_input_file)
        print(f"构象 {idx} 的 Gaussian 输入文件已保存到: {gaussian_input_file}")


    # 将所有结果写入到一个汇总文件中
    summary_file = os.path.join(output_dir, "optimization_summary.txt")
    with open(summary_file, "w") as f:
        f.write("Conformer Index\tOptimization Result Code\tOptimized Energy (kcal/mol)\n")
        for result in all_results:
            f.write(f"{result['conformer_index']}\t{result['optimization_result']}\t{result['optimized_energy']:.4f}\n")

    print(f"所有构象的优化结果已汇总到: {summary_file}")



if __name__ == '__main__':
    main()