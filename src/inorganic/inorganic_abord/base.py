from transform.inorganic.feature import get_atom_features, get_bond_features, get_bond_index, get_mol_features
import rdkit.Chem
import torch
from rdkit.Chem import Atom, Bond, Mol, AllChem
from torch_geometric.data import Data as PyGData
from typing import Any, Optional, Union, Dict, Tuple, List, Callable
from torch_geometric.typing import Adj
from utils.typing import NodeVector, EdgeVector, GlobalVector, SMILES
from utils.common import check_and_convert_to_rdmol

class MolGraphTransform:
    atom_feature_fn: Callable[[Atom], List[float]] = staticmethod(get_atom_features)
    bond_feature_fn: Optional[Callable[[Bond], List[float]]] = staticmethod(get_bond_features)
    bond_index_fn: Optional[Callable[[Mol], Tuple[List[Bond], Adj]]] = staticmethod(get_bond_index)
    mol_feature_fn: Optional[Callable[[Mol], List[float]]] = staticmethod(get_mol_features)

    def __call__(self, mol: SMILES | Mol) -> PyGData:
        return self.call(mol)

    @classmethod
    def call(cls, mol: SMILES | Mol) -> PyGData:
        return PyGData(**cls.processing(mol))

    @classmethod
    def processing(cls, mol: SMILES | Mol) -> Dict:
        rdmol = check_and_convert_to_rdmol(mol)
        retval = {'x': cls.get_atom_feature(rdmol)}

        if cls.bond_index_fn is not None:
            bonds, bond_index = cls.get_bond_index(rdmol)
            retval['edge_index'] = bond_index
            if cls.bond_feature_fn is not None:
                retval['edge_attr'] = cls.get_bond_feature(bonds)
        else:
            retval['edge_index'] = torch.zeros((2, 0), dtype=torch.long)

        if cls.mol_feature_fn is not None:
            retval['global_x'] = cls.get_mol_feature(rdmol)

        # 添加分子坐标
        # retval['pos'] = cls.get_mol_pos(rdmol)

        return retval

    @classmethod
    def get_atom_feature(cls, rdmol: Mol) -> NodeVector:
        print('this')
        return torch.FloatTensor([cls.atom_feature_fn(atom) for atom in rdmol.GetAtoms()])

    @classmethod
    def get_bond_index(cls, rdmol: Mol) -> Tuple[List[Bond], Adj]:
        return cls.bond_index_fn(rdmol)

    @classmethod
    def get_bond_feature(cls, bonds: List[Bond]) -> EdgeVector:
        try:
            return torch.FloatTensor([cls.bond_feature_fn(bond) for bond in bonds])
        except Exception as e:
            raise ValueError(f"Failed to extract bond features: {e}")

    @classmethod
    def get_mol_feature(cls, rdmol: Mol) -> GlobalVector:
        return torch.FloatTensor(cls.mol_feature_fn(rdmol)).unsqueeze(0)

    @classmethod
    def get_mol_pos(cls, rdmol: Mol) -> torch.Tensor:
        rdmol = rdkit.Chem.AddHs(rdmol)
        AllChem.EmbedMolecule(rdmol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(rdmol)
        conf = rdmol.GetConformer()
        pos = torch.tensor([list(conf.GetAtomPosition(i)) for i in range(rdmol.GetNumAtoms())], dtype=torch.float32)
        return pos