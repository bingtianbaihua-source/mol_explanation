from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional, List, Tuple
import torch
from torch_geometric.data import Data as PyGData


class AtomType(Enum):
    """原子类型枚举（支持扩展）"""
    METAL = auto()      # 金属原子（如 Na, K, Cs）
    OXYGEN = auto()     # 氧原子
    CARBON = auto()     # 碳原子
    PHOSPHORUS = auto() # 磷原子
    OTHER = auto()      # 其他原子

@dataclass
class InorganicAtom:
    idx: int             # 原子索引
    symbol: str          # 原子符号（如 "Na", "O"）
    atom_type: AtomType  # 原子类型
    formal_charge: int   # 形式电荷
    oxidation_state: int # 氧化态
    en: float            # 电负性（缩放后）
    mass: float          # 原子质量（缩放后）

    @classmethod
    def from_symbol(cls, idx: int, symbol: str, charge: int = 0, oxidation: Optional[int] = None):
        """通过原子符号创建原子对象"""
        # 原子类型映射
        type_map = {
            "Na": AtomType.METAL, "K": AtomType.METAL, "Cs": AtomType.METAL,
            "O": AtomType.OXYGEN, "C": AtomType.CARBON, "P": AtomType.PHOSPHORUS
        }
        # 电负性字典（实际值需要扩展）
        en_data = {"Na": 0.93, "K": 0.82, "Cs": 0.79, "O": 3.44, "C": 2.55, "P": 2.19}
        # 原子质量字典（实际值需要扩展）
        mass_data = {"Na": 22.99, "K": 39.10, "Cs": 132.91, "O": 16.00, "C": 12.01, "P": 30.97}

        return cls(
            idx=idx,
            symbol=symbol,
            atom_type=type_map.get(symbol, AtomType.OTHER),
            formal_charge=charge,
            oxidation_state=oxidation if oxidation is not None else charge,
            en=en_data.get(symbol, 0.0) * 0.25,  # 缩放电负性
            mass=mass_data.get(symbol, 0.0) * 0.01  # 缩放质量
        )
    
class BondType(Enum):
    """键类型枚举（支持扩展）"""
    IONIC = auto()      # 离子键（如 Na-O）
    COVALENT = auto()   # 共价键（如 C-O）
    DOUBLE = auto()     # 双键（如 P=O）
    OTHER = auto()      # 其他键类型

@dataclass
class InorganicBond:
    bond_type: BondType     # 键类型
    src_atom_idx: int       # 起始原子索引
    dst_atom_idx: int       # 终止原子索引
    length: Optional[float] = None  # 键长（可选）

class InorganicMolecule:
    def __init__(self):
        self.atoms: List[InorganicAtom] = []
        self.bonds: List[InorganicBond] = []

    def add_atom(self, symbol: str, charge: int = 0, oxidation: Optional[int] = None):
        """添加原子"""
        idx = len(self.atoms)
        self.atoms.append(InorganicAtom.from_symbol(idx, symbol, charge, oxidation))

    def add_bond(self, src_idx: int, dst_idx: int, bond_type: BondType):
        """添加键"""
        self.bonds.append(InorganicBond(bond_type, src_idx, dst_idx))

    def get_atom_features(self) -> torch.Tensor:
        """原子特征矩阵 [num_atoms, num_features]"""
        features = []
        for atom in self.atoms:
            # 特征设计示例：类型、电荷、氧化态、电负性、质量
            feat = [
                atom.atom_type.value,
                atom.formal_charge,
                atom.oxidation_state,
                atom.en,
                atom.mass
            ]
            features.append(feat)
        return torch.tensor(features, dtype=torch.float32)

    def get_bond_features(self) -> torch.Tensor:
        """键特征矩阵 [num_bonds, num_features]"""
        features = []
        for bond in self.bonds:
            # 特征设计示例：键类型、是否离子键、是否双键
            feat = [
                bond.bond_type.value,
                1 if bond.bond_type == BondType.IONIC else 0,
                1 if bond.bond_type == BondType.DOUBLE else 0
            ]
            features.append(feat)
        return torch.tensor(features, dtype=torch.float32)

    def get_bond_index(self) -> Tuple[List[InorganicBond], torch.Tensor]:
        """
        获取键列表和边索引矩阵。
        :return: 键列表和边索引矩阵 [2, num_bonds * 2]
        """
        if len(self.bonds) > 0:
            # 生成边索引矩阵 [2, num_bonds]
            edge_index = [[bond.src_atom_idx, bond.dst_atom_idx] for bond in self.bonds]
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            # 添加反向边
            edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)  # [2, num_bonds * 2]
            
            # 复制键列表以匹配边索引
            bonds = self.bonds * 2
            return bonds, edge_index
        else:
            # 如果没有键，返回空列表和空的边索引矩阵
            return [], torch.zeros((2, 0), dtype=torch.long)
    
    def get_mol_features(self) -> torch.Tensor:
        """分子全局特征（示例：总电荷、平均电负性）"""
        total_charge = sum(atom.formal_charge for atom in self.atoms)
        avg_en = sum(atom.en for atom in self.atoms) / len(self.atoms)
        return torch.tensor([total_charge, avg_en], dtype=torch.float32).unsqueeze(0)

class MolGraphTransform():

    def __call__(self, iomol: InorganicMolecule) -> PyGData:
        return self.call(iomol)
    
    @classmethod
    def call(cls, iomol: InorganicMolecule) -> PyGData:
        return PyGData(**cls.processing(iomol))
    
    @classmethod
    def processing(cls, iomol: InorganicMolecule) -> Dict:

        retval = {}

        retval['x'] = iomol.get_atom_features()

        _, bond_index = iomol.get_bond_index()

        retval['edge_index'] = bond_index

        retval['edge_attr'] = iomol.get_bond_features()

        retval['global_x'] = iomol.get_mol_features()

        return retval
