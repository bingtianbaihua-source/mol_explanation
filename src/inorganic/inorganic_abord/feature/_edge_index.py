import torch
from rdkit import Chem
from rdkit.Chem import Mol, Bond
from torch_geometric.typing import Adj
from typing import List, Tuple

__all__ = ['get_bonds_normal', 'get_bonds_complete']

def get_bonds_normal(mol: Mol) -> Tuple[List[Bond], Adj]:
    """
    获取无机物分子中实际存在的键及其索引。
    :param mol: RDKit 的 Mol 对象
    :return: 键列表和键索引（Adjacency Matrix）
    """
    bonds = list(mol.GetBonds())
    if len(bonds) > 0:
        bond_index = torch.LongTensor([(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in bonds]).T
        bond_index = torch.cat([bond_index, bond_index[[1, 0]]], dim=1)  # (2, E), E = NumRealBonds
        bonds = bonds * 2  # 由于无向图，每条边需要存储两次
        return bonds, bond_index
    else:
        return [], torch.zeros((2, 0), dtype=torch.long)

def get_bonds_complete(mol: Mol) -> Tuple[List[Bond], Adj]:
    """
    获取无机物分子中所有原子之间的键（包括虚拟边）。
    :param mol: RDKit 的 Mol 对象
    :return: 键列表和键索引（Adjacency Matrix）
    """
    num_atoms = mol.GetNumAtoms()
    if num_atoms > 1:
        bond_index = [[i, j] for i in range(num_atoms) for j in range(num_atoms) if i != j]
        bonds = [mol.GetBondBetweenAtoms(i, j) for i, j in bond_index]  # None for virtual(unreal) edge
        bond_index = torch.LongTensor(bond_index).T  # (2, E), E = V(V-1)
        return bonds, bond_index
    else:
        return [], torch.zeros((2, 0), dtype=torch.long)

def is_ionic_bond(bond: Bond) -> bool:
    """
    判断是否为离子键。
    :param bond: RDKit 的 Bond 对象
    :return: 是否为离子键
    """
    atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
    electronegativity_diff = abs(atom1.GetDoubleProp("EN") - atom2.GetDoubleProp("EN"))
    return electronegativity_diff > 1.7  # 电负性差大于 1.7 可能是离子键

def is_metallic_bond(bond: Bond) -> bool:
    """
    判断是否为金属键。
    :param bond: RDKit 的 Bond 对象
    :return: 是否为金属键
    """
    atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
    return atom1.GetSymbol() in {"Na", "Mg", "Al", "Fe", "Cu", "Zn"} and \
           atom2.GetSymbol() in {"Na", "Mg", "Al", "Fe", "Cu", "Zn"}

def is_coordinate_bond(bond: Bond) -> bool:
    """
    判断是否为配位键。
    :param bond: RDKit 的 Bond 对象
    :return: 是否为配位键
    """
    atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
    metal_symbols = {"Na", "Mg", "Al", "Fe", "Cu", "Zn"}
    non_metal_symbols = {"O", "N", "Cl", "F", "S"}
    return (atom1.GetSymbol() in metal_symbols and atom2.GetSymbol() in non_metal_symbols) or \
           (atom2.GetSymbol() in metal_symbols and atom1.GetSymbol() in non_metal_symbols)