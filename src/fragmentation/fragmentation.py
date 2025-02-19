from rdkit import Chem
from fragmentation.utils import *
from utils.typing import *
from itertools import combinations
import random
import logging
import torch

from utils.common import convert_to_rdmol, convert_to_SMILES


class Unit:
    def __init__(self, graph, atom_indices: tuple[int]):
        self.graph = graph
        atom_indices_set = set(atom_indices)
        bond_indices = []
        for bond in graph.rdmol.GetBonds():
            atom_idx1, atom_idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if atom_idx1 in atom_indices_set and atom_idx2 in atom_indices_set:
                bond_indices.append(bond.GetIdx())

        self.atom_indices = atom_indices
        self.bond_indices = tuple(bond_indices)

        self.neighbors = []
        self.connections = []

    def add_connection(self, neighbor_unit, connection):
        self.neighbors.append(neighbor_unit)
        self.connections.append(connection)

    def to_rdmol(self):
        return self.graph.get_submol([self])

    def to_fragment(self, connection):
        assert connection in self.connections
        atom_map = {}
        submol = self.graph.get_submol([self], atom_map=atom_map)
        if self == connection.units[0]:
            atom_index = atom_map[connection.atom_indices[0]]
        else:
            atom_index = atom_map[connection.atom_indices[1]]
        bondtype = connection.bondtype

        rwmol = Chem.RWMol(submol)
        add_dummy_atom(rwmol, atom_index, bondtype)
        fragment = rwmol.GetMol()
        fragment = Chem.MolFromSmiles(Chem.MolToSmiles(fragment))
        return fragment


class Connection:
    def __init__(self, unit1: Unit, unit2: Unit, atom_index1: int, atom_index2: int, bond_index: int, bondtype: BondType):
        self.units = (unit1, unit2)
        self.atom_indices = (atom_index1, atom_index2)
        self.bond_index = bond_index
        self._bondtype = int(bondtype)
        unit1.add_connection(unit2, self)
        unit2.add_connection(unit1, self)

    @property
    def bondtype(self):
        return BondType.values[self._bondtype]


class FragmentedGraph:
    def fragmentation(self, mol: Mol):
        raise NotImplementedError

    def __init__(self, mol: SMILES | Mol):
        rdmol = convert_to_rdmol(mol)
        self.rdmol = rdmol

        units, connections = self.fragmentation(rdmol)
        self.units = units
        self.num_units = len(units)

        self.connections = connections
        self.connection_dict = {}
        for connection in connections:
            unit1, unit2 = connection.units
            self.connection_dict[(unit1, unit2)] = connection
            self.connection_dict[(unit2, unit1)] = connection

    def __len__(self):
        return self.num_units

    def get_submol(self, unit_list: list[Unit], atom_map: dict):
        atom_indices = []
        bond_indices = []
        for unit in unit_list:
            atom_indices += unit.atom_indices
            bond_indices += unit.bond_indices

        for unit1, unit2 in combinations(unit_list, 2):
            connection = self.connection_dict.get((unit1, unit2), None)
            if connection is not None:
                bond_indices.append(connection.bond_index)

        atom_map.update({atom_index: new_atom_index for new_atom_index, atom_index in enumerate(atom_indices)})

        rwmol = Chem.RWMol()
        src_atom_list = [self.rdmol.GetAtomWithIdx(atom_index) for atom_index in atom_indices]
        src_bond_list = [self.rdmol.GetBondWithIdx(bond_index) for bond_index in bond_indices]
        for src_atom in src_atom_list:
            rwmol.AddAtom(src_atom)

        for src_bond in src_bond_list:
            src_atom_index1, src_atom_index2 = src_bond.GetBeginAtomIdx(), src_bond.GetEndAtomIdx()
            dst_atom_index1, dst_atom_index2 = atom_map[src_atom_index1], atom_map[src_atom_index2]
            bondtype = src_bond.GetBondType()
            rwmol.AddBond(dst_atom_index1, dst_atom_index2, bondtype)

        for src_atom, dst_atom in zip(src_atom_list, rwmol.GetAtoms()):
            if dst_atom.GetAtomicNum() == 7:
                degree_diff = src_atom.GetDegree() - dst_atom.GetDegree()
                if degree_diff > 0:
                    dst_atom.SetNumExplicitHs(dst_atom.GetNumExplicitHs() + degree_diff)

        submol = rwmol.GetMol()
        Chem.SanitizeMol(submol)

        return submol

    def get_datapoint(self, traj=None):
        if traj is None:
            traj = self.get_subtrajectory(min_length=2)
        scaffold_units, fragment_unit = traj[:-1], traj[-1]

        if fragment_unit is None:
            scaffold = Chem.Mol(self.rdmol)
            return scaffold, None, (None, None)
        else:
            neighbor_units = set(fragment_unit.neighbors).intersection(set(scaffold_units))
            assert len(neighbor_units) == 1
            neighbor_unit = neighbor_units.pop()
            connection = self.connection_dict[(fragment_unit, neighbor_unit)]

            atom_map = {}
            scaffold = self.get_submol(scaffold_units, atom_map=atom_map)
            fragment = fragment_unit.to_fragment(connection)

            if fragment_unit is connection.units[0]:
                scaffold_atom_index = atom_map[connection.atom_indices[1]]
            else:
                scaffold_atom_index = atom_map[connection.atom_indices[0]]
            fragment_atom_index = fragment.GetNumAtoms() - 1

            return scaffold, fragment, (scaffold_atom_index, fragment_atom_index)

    def get_subtrajectory(self, length=None, min_length=1, max_length=None):
        if length is None:
            assert max_length is None or max_length >= min_length
            if max_length is None:
                max_length = self.num_units + 1
            else:
                max_length = min(max_length, self.num_units + 1)
            length = random.randrange(min_length, max_length + 1)

        if length == self.num_units + 1:
            traj = list(self.units) + [None]
        else:
            traj: list[Unit] = []
            neighbors: set[Unit] = set()
            traj_length = 0
            while True:
                if traj_length == 0:
                    unit = random.choice(self.units)
                else:
                    unit = random.choice(list(neighbors))
                traj.append(unit)
                traj_length += 1
                if traj_length == length:
                    break
                neighbors.update(unit.neighbors)
                neighbors.difference_update(traj)
        return traj


def fragmentation(mol: SMILES | Mol):
    return FragmentedGraph(mol)


class Fragmentation:
    fragmentation = staticmethod(fragmentation)

    def __call__(self, mol: SMILES | Mol):
        return self.fragmentation(mol)

    @staticmethod
    def merge(scaffold: Mol, fragment: Mol, scaffold_atom_index, fragment_atom_index):
        merge(scaffold, fragment, scaffold_atom_index, fragment_atom_index)

    @classmethod
    def decompose(cls, mol: SMILES | Mol):
        rdmol = convert_to_rdmol(mol)
        fragmented_mol = cls.fragmentation(rdmol)
        if len(fragmented_mol) == 1:
            fragments = []
        else:
            fragments = [Chem.MolToSmiles(unit.to_fragment(connection))
                         for unit in fragmented_mol.units
                         for connection in unit.connections]
        return fragments

class BlockLibrary:
    fragmentation = Fragmentation()

    def __init__(self,
                 library_path: str=None,
                 smiles_list: list[SMILES]=None,
                 frequency_list: FloatTensor = None,
                 use_frequency: bool=True,
                 save_rdmol: bool=False):
        if library_path is not None:
            smiles_list,frequency_list = self.load_library_file(library_path, use_frequency)

        assert smiles_list is not None
        if not use_frequency:
            frequency_list = None

        self._smilies_list = smiles_list

        if frequency_list is not None:
            self._frequency_distribution = frequency_list
        else:
            if use_frequency:
                logging.warning(f'No Frequency Information in library')
            self._frequency_distribution = torch.full((len(smiles_list),), 1 / len(smiles_list))
            
        self._smilies_to_index = {smiles: index for index, smiles in enumerate(smiles_list)}

        if save_rdmol:
            self._rdmol_list = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        else:
            self._rdmol_list = None

    def __len__(self):
        return len(self._smilies_list)
    
    def __getitem__(self, index: int):
        return self._smilies_list[index]
    
    def get_smiles(self, index: int):
        return self._smilies_list[index]
    
    def get_rdmol(self, index: int):
        if self._rdmol_list is not None:
            return self._rdmol_list[index]
        else:
            return Chem.MolFromSmiles(self.get_smiles(index))
        
    def get_index(self, mol: SMILES|Mol):
        smi = convert_to_SMILES(mol)
        return self._smilies_to_index[smi]
    
    @property
    def smiles_list(self):
        return self._smilies_list
    
    @property
    def rdmol_list(self):
        if self._rdmol_list is not None:
            return self._rdmol_list
        else:
            return [Chem.MolFromSmiles(smi) for smi in self._smilies_list]
        
    