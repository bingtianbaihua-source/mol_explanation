from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from typing import Optional, List
from rdkit.Chem import Bond

__all__ = ['get_bond_features', 'NUM_BOND_FEATURES']

### 定义无机物的键类型 ###
INORGANIC_BOND_TYPES = {
    BondType.SINGLE: 0,      # 单键
    BondType.DOUBLE: 1,      # 双键
    BondType.TRIPLE: 2,      # 三键
    "IONIC": 3,              # 离子键
    "METALLIC": 4,           # 金属键
    "COORDINATE": 5,         # 配位键
    BondType.OTHER: 6,       # 其他类型
}
NUM_BOND_FEATURES = len(INORGANIC_BOND_TYPES)

def get_bond_features(bond: Optional[Bond]) -> List[int]:
    """
    获取无机物键的特征向量。
    :param bond: RDKit 的 Bond 对象
    :return: 键的特征向量（one-hot 编码）
    """
    retval = [0] * NUM_BOND_FEATURES  # 初始化 one-hot 编码列表
    if bond is None:
        return retval
    
    # 获取键类型
    bond_type = bond.GetBondType()
    
    # 判断键类型并设置 one-hot 编码
    if bond_type == BondType.SINGLE:
        retval[INORGANIC_BOND_TYPES[BondType.SINGLE]] = 1
    elif bond_type == BondType.DOUBLE:
        retval[INORGANIC_BOND_TYPES[BondType.DOUBLE]] = 1
    elif bond_type == BondType.TRIPLE:
        retval[INORGANIC_BOND_TYPES[BondType.TRIPLE]] = 1
    else:
        # 处理无机物的键类型
        if is_ionic_bond(bond):  # 自定义函数判断是否为离子键
            retval[INORGANIC_BOND_TYPES["IONIC"]] = 1
        elif is_metallic_bond(bond):  # 自定义函数判断是否为金属键
            retval[INORGANIC_BOND_TYPES["METALLIC"]] = 1
        elif is_coordinate_bond(bond):  # 自定义函数判断是否为配位键
            retval[INORGANIC_BOND_TYPES["COORDINATE"]] = 1
        else:
            retval[INORGANIC_BOND_TYPES[BondType.OTHER]] = 1
    
    return retval

def is_ionic_bond(bond: Bond) -> bool:
    """
    判断是否为离子键。
    :param bond: RDKit 的 Bond 对象
    :return: 是否为离子键
    """
    atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
    electronegativity_diff = abs(atom1.GetElectronegativity() - atom2.GetElectronegativity())
    return electronegativity_diff > 1.7  # 电负性差大于 1.7 可能是离子键

def is_metallic_bond(bond: Bond) -> bool:
    """
    判断是否为金属键。
    :param bond: RDKit 的 Bond 对象
    :return: 是否为金属键
    """
    # 金属键的判断通常需要结合晶体结构信息
    # 这里简单判断是否为金属元素之间的键
    atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
    return atom1.GetSymbol() in {"Na", "Mg", "Al", "Fe", "Cu", "Zn"} and \
           atom2.GetSymbol() in {"Na", "Mg", "Al", "Fe", "Cu", "Zn"}

def is_coordinate_bond(bond: Bond) -> bool:
    """
    判断是否为配位键。
    :param bond: RDKit 的 Bond 对象
    :return: 是否为配位键
    """
    # 配位键的判断通常需要结合配位化合物的信息
    # 这里简单判断是否为金属与非金属之间的键
    atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
    metal_symbols = {"Na", "Mg", "Al", "Fe", "Cu", "Zn"}
    non_metal_symbols = {"O", "N", "Cl", "F", "S"}
    return (atom1.GetSymbol() in metal_symbols and atom2.GetSymbol() in non_metal_symbols) or \
           (atom2.GetSymbol() in metal_symbols and atom1.GetSymbol() in non_metal_symbols)