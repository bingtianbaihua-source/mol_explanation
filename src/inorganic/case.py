from Inorganic import *

# from inorganic.Inorganic import *


def build_cs2co3() -> InorganicMolecule:
    mol = InorganicMolecule()
    # 添加原子（Cs: +1, C: +4, O: -2）
    mol.add_atom("Cs", charge=1, oxidation=1)
    mol.add_atom("Cs", charge=1, oxidation=1)
    mol.add_atom("C", charge=4, oxidation=4)
    mol.add_atom("O", charge=-2, oxidation=-2)
    mol.add_atom("O", charge=-2, oxidation=-2)
    mol.add_atom("O", charge=-2, oxidation=-2)
    
    # 添加键（C-O 双键）
    mol.add_bond(2, 3, BondType.DOUBLE)  # C=O
    mol.add_bond(2, 4, BondType.DOUBLE)  # C=O
    mol.add_bond(2, 5, BondType.DOUBLE)  # C=O
    
    # 添加离子键（Cs与CO3^2-之间的离子键）
    mol.add_bond(0, 3, BondType.IONIC)  # Cs-O
    mol.add_bond(1, 4, BondType.IONIC)  # Cs-O
    return mol

def build_naome() -> InorganicMolecule:
    mol = InorganicMolecule()
    # 添加原子（Na: +1, O: -1, C: 0）
    mol.add_atom("Na", charge=1, oxidation=1)  # Na
    mol.add_atom("O", charge=-1, oxidation=-2)  # O
    mol.add_atom("C", charge=0, oxidation=0)  # C

    # 添加键
    mol.add_bond(0, 1, BondType.IONIC)  # Na-O 离子键
    mol.add_bond(1, 2, BondType.COVALENT)  # O-C 共价键
    return mol

def build_k3po4() -> InorganicMolecule:
    mol = InorganicMolecule()
    # 添加原子（K: +1, P: +5, O: -2）
    mol.add_atom("K", charge=1, oxidation=1)  # K1
    mol.add_atom("K", charge=1, oxidation=1)  # K2
    mol.add_atom("K", charge=1, oxidation=1)  # K3
    mol.add_atom("P", charge=5, oxidation=5)  # P
    mol.add_atom("O", charge=-2, oxidation=-2)  # O1
    mol.add_atom("O", charge=-2, oxidation=-2)  # O2
    mol.add_atom("O", charge=-2, oxidation=-2)  # O3
    mol.add_atom("O", charge=-2, oxidation=-2)  # O4

    # 添加键
    mol.add_bond(0, 4, BondType.IONIC)  # K1-O1 离子键
    mol.add_bond(1, 5, BondType.IONIC)  # K2-O2 离子键
    mol.add_bond(2, 6, BondType.IONIC)  # K3-O3 离子键
    mol.add_bond(3, 4, BondType.DOUBLE)  # P=O1 双键
    mol.add_bond(3, 5, BondType.COVALENT)  # P-O2 单键
    mol.add_bond(3, 6, BondType.COVALENT)  # P-O3 单键
    mol.add_bond(3, 7, BondType.COVALENT)  # P-O4 单键
    return mol

Inorganic_map = {
    'Cs2CO3': build_cs2co3,
    'NaOMe': build_naome,
    'K3PO4': build_k3po4,
}

if __name__ == '__main__':
    cs2co3 = Inorganic_map["Cs2CO3"]()
    atom_features = cs2co3.get_atom_features()      # 原子特征矩阵
    bond_features = cs2co3.get_bond_features()      # 键特征矩阵
    edge_index = cs2co3.get_bond_index()            # 边索引
    mol_features = cs2co3.get_mol_features()        # 分子全局特征

    # 输出形状
    print(f"Atom features shape: {atom_features.shape}")  # [6, 5]
    print(f"Bond features shape: {bond_features.shape}")  # [5, 3]
    print(f"Edge index shape: {edge_index.shape}")        # [2, 5]
    print(f"Mol features shape: {mol_features.shape}")    # [1, 2]