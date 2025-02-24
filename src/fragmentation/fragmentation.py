from rdkit import Chem
from fragmentation.utils import *
from utils.typing import *
from itertools import combinations
from typing import List, Optional, Tuple, Set, Dict, Union
import random
import logging
import torch
import os
from utils.common import convert_to_rdmol, convert_to_SMILES


class Unit:
    def __init__(self, graph, atom_indices: Tuple[int]):
        self.graph = graph
        atom_indices_set = set(atom_indices)
        bond_indices = [
            bond.GetIdx() for bond in graph.rdmol.GetBonds()
            if bond.GetBeginAtomIdx() in atom_indices_set and bond.GetEndAtomIdx() in atom_indices_set
        ]

        self.atom_indices = atom_indices
        self.bond_indices = tuple(bond_indices)
        self.neighbors: List[Unit] = []
        self.connections: List[Connection] = []

    def add_connection(self, neighbor_unit: 'Unit', connection: 'Connection'):
        self.neighbors.append(neighbor_unit)
        self.connections.append(connection)

    def to_rdmol(self) -> Chem.Mol:
        return self.graph.get_submol([self])

    def to_fragment(self, connection: 'Connection') -> Chem.Mol:
        assert connection in self.connections, "Connection must be associated with this unit"
        atom_map = {}
        submol = self.graph.get_submol([self], atom_map=atom_map)
        atom_index = atom_map[connection.atom_indices[0 if self == connection.units[0] else 1]]
        bondtype = connection.bondtype

        rwmol = Chem.RWMol(submol)
        add_dummy_atom(rwmol, atom_index, bondtype)
        fragment = rwmol.GetMol()
        return Chem.MolFromSmiles(Chem.MolToSmiles(fragment))


class Connection:
    def __init__(
        self,
        unit1: Unit,
        unit2: Unit,
        atom_index1: int,
        atom_index2: int,
        bond_index: int,
        bondtype: BondType
    ):
        self.units = (unit1, unit2)
        self.atom_indices = (atom_index1, atom_index2)
        self.bond_index = bond_index
        self._bondtype = int(bondtype)
        unit1.add_connection(unit2, self)
        unit2.add_connection(unit1, self)

    @property
    def bondtype(self) -> BondType:
        return BondType.values[self._bondtype]


class FragmentedGraph:
    def fragmentation(self, mol: Chem.Mol) -> Tuple[List[Unit], List[Connection]]:
        raise NotImplementedError

    def __init__(self, mol: Union[str, Chem.Mol]):
        self.rdmol = convert_to_rdmol(mol)
        self.units, self.connections = self.fragmentation(self.rdmol)
        self.num_units = len(self.units)
        self.connection_dict = self._build_connection_dict(self.connections)

    def _build_connection_dict(self, connections: List[Connection]) -> Dict[Tuple[Unit, Unit], Connection]:
        connection_dict = {}
        for connection in connections:
            unit1, unit2 = connection.units
            connection_dict[(unit1, unit2)] = connection
            connection_dict[(unit2, unit1)] = connection
        return connection_dict

    def __len__(self) -> int:
        return self.num_units

    def get_submol(self, unit_list: List[Unit], atom_map: Optional[Dict[int, int]] = None) -> Chem.Mol:
        if atom_map is None:
            atom_map = {}

        atom_indices = []
        bond_indices = []
        for unit in unit_list:
            atom_indices.extend(unit.atom_indices)
            bond_indices.extend(unit.bond_indices)

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
            rwmol.AddBond(dst_atom_index1, dst_atom_index2, src_bond.GetBondType())

        for src_atom, dst_atom in zip(src_atom_list, rwmol.GetAtoms()):
            if dst_atom.GetAtomicNum() == 7:
                degree_diff = src_atom.GetDegree() - dst_atom.GetDegree()
                if degree_diff > 0:
                    dst_atom.SetNumExplicitHs(dst_atom.GetNumExplicitHs() + degree_diff)

        submol = rwmol.GetMol()
        Chem.SanitizeMol(submol)
        return submol

    def get_datapoint(self, traj: Optional[List[Unit]] = None) -> Tuple[Chem.Mol, Optional[Chem.Mol], Tuple[Optional[int], Optional[int]]]:
        if traj is None:
            traj = self.get_subtrajectory(min_length=2)
        scaffold_units, fragment_unit = traj[:-1], traj[-1]

        if fragment_unit is None:
            return Chem.Mol(self.rdmol), None, (None, None)

        neighbor_units = set(fragment_unit.neighbors).intersection(set(scaffold_units))
        assert len(neighbor_units) == 1, "Fragment unit must have exactly one neighbor in the scaffold"
        neighbor_unit = neighbor_units.pop()
        connection = self.connection_dict[(fragment_unit, neighbor_unit)]

        atom_map = {}
        scaffold = self.get_submol(scaffold_units, atom_map=atom_map)
        fragment = fragment_unit.to_fragment(connection)

        scaffold_atom_index = atom_map[connection.atom_indices[1 if fragment_unit is connection.units[0] else 0]]
        fragment_atom_index = fragment.GetNumAtoms() - 1

        return scaffold, fragment, (scaffold_atom_index, fragment_atom_index)

    def get_subtrajectory(
        self,
        length: Optional[int] = None,
        min_length: int = 1,
        max_length: Optional[int] = None
    ) -> List[Optional[Unit]]:
        if length is None:
            if max_length is None:
                max_length = self.num_units + 1
            else:
                max_length = min(max_length, self.num_units + 1)
            length = random.randrange(min_length, max_length + 1)

        if length == self.num_units + 1:
            return list(self.units) + [None]

        traj: List[Unit] = []
        neighbors: Set[Unit] = set()
        while len(traj) < length:
            unit = random.choice(list(neighbors)) if neighbors else random.choice(self.units)
            traj.append(unit)
            neighbors.update(unit.neighbors)
            neighbors.difference_update(traj)
        return traj


def fragmentation(mol: Union[str, Chem.Mol]) -> FragmentedGraph:
    return FragmentedGraph(mol)


class Fragmentation:
    fragmentation = staticmethod(fragmentation)

    def __call__(self, mol: Union[str, Chem.Mol]) -> FragmentedGraph:
        return self.fragmentation(mol)

    @staticmethod
    def merge(scaffold: Chem.Mol, fragment: Chem.Mol, scaffold_atom_index: int, fragment_atom_index: int):
        merge(scaffold, fragment, scaffold_atom_index, fragment_atom_index)

    @classmethod
    def decompose(cls, mol: Union[str, Chem.Mol]) -> List[str]:
        rdmol = convert_to_rdmol(mol)
        fragmented_mol = cls.fragmentation(rdmol)
        if len(fragmented_mol) == 1:
            return []
        return [
            Chem.MolToSmiles(unit.to_fragment(connection))
            for unit in fragmented_mol.units
            for connection in unit.connections
        ]

class BlockLibrary:
    fragmentation = Fragmentation()

    def __init__(
        self,
        library_path: Optional[str] = None,
        smiles_list: Optional[List[SMILES]] = None,
        frequency_list: Optional[FloatTensor] = None,
        use_frequency: bool = True,
        save_rdmol: bool = False
    ):
        if library_path is not None:
            smiles_list, frequency_list = self.load_library_file(library_path, use_frequency)

        assert smiles_list is not None, "smiles_list must be provided or loaded from library_path"

        self._smiles_list = smiles_list
        self._frequency_distribution = self._initialize_frequency_distribution(frequency_list, use_frequency)
        self._smiles_to_index = {smiles: index for index, smiles in enumerate(smiles_list)}
        self._rdmol_list = self._initialize_rdmol_list(smiles_list, save_rdmol)

    def _initialize_frequency_distribution(
        self,
        frequency_list: Optional[FloatTensor],
        use_frequency: bool
    ) -> FloatTensor:
        if frequency_list is not None:
            return frequency_list
        elif use_frequency:
            logging.warning('No Frequency Information in library')
        return torch.full((len(self._smiles_list),), 1 / len(self._smiles_list))

    def _initialize_rdmol_list(
        self,
        smiles_list: List[SMILES],
        save_rdmol: bool
    ) -> Optional[List[Mol]]:
        return [Chem.MolFromSmiles(smi) for smi in smiles_list] if save_rdmol else None

    def __len__(self) -> int:
        return len(self._smiles_list)

    def __getitem__(self, index: int) -> SMILES:
        return self._smiles_list[index]

    def get_smiles(self, index: int) -> SMILES:
        return self._smiles_list[index]

    def get_rdmol(self, index: int) -> Mol:
        if self._rdmol_list is not None:
            return self._rdmol_list[index]
        return Chem.MolFromSmiles(self.get_smiles(index))

    def get_index(self, mol: Union[SMILES, Mol]) -> int:
        smi = convert_to_SMILES(mol)
        return self._smiles_to_index[smi]

    @property
    def smiles_list(self) -> List[SMILES]:
        return self._smiles_list

    @property
    def rdmol_list(self) -> List[Mol]:
        if self._rdmol_list is not None:
            return self._rdmol_list
        return [Chem.MolFromSmiles(smi) for smi in self._smiles_list]

    @property
    def frequency_distribution(self) -> FloatTensor:
        return self._frequency_distribution

    def load_library_file(
        self,
        library_path: str,
        use_frequency: bool = True
    ) -> tuple[List[SMILES], Optional[FloatTensor]]:
        extension = os.path.splitext(library_path)[1]
        assert extension in ['.smi', '.csv'], "Unsupported file extension"

        with open(library_path) as f:
            if extension == '.smi':
                smiles_list = [line.strip() for line in f]
                frequency_list = None
            else:
                header = f.readline().strip().split(',')
                if len(header) == 1:
                    smiles_list = [line.strip() for line in f]
                    frequency_list = None
                else:
                    lines = [line.strip().split(',') for line in f]
                    smiles_list = [smiles for smiles, _ in lines]
                    frequency_list = torch.FloatTensor([float(freq) for _, freq in lines]) if use_frequency else None

        return smiles_list, frequency_list

    @classmethod
    def create_library_file(
        cls,
        library_path: str,
        mol_list: List[Union[SMILES, Mol]],
        save_frequency: bool = True,
        cpus: int = 1
    ) -> List[bool]:
        from collections import Counter
        import parmap

        extension = os.path.splitext(library_path)[1]
        assert extension in ['.smi', '.csv'], "Unsupported file extension"

        res = parmap.map(cls.decompose, mol_list, pm_processes=cpus, pm_chunksize=1000, pm_pbar=True)
        block_list = []
        flag_list = []
        for blocks in res:
            if blocks is None:
                flag_list.append(False)
            else:
                flag_list.append(True)
                block_list.extend(blocks)

        block_freq_list = sorted(Counter(block_list).items(), key=lambda item: item[1], reverse=True)

        with open(library_path, 'w') as w:
            if save_frequency and extension == '.csv':
                w.write('SMILES,frequency\n')
                for block, freq in block_freq_list:
                    block = convert_to_SMILES(block)
                    w.write(f'{block},{freq}\n')
            else:
                if extension == '.csv':
                    w.write('SMILES\n')
                for block, _ in block_freq_list:
                    block = convert_to_SMILES(block)
                    w.write(f'{block}\n')

        return flag_list

    @classmethod
    def decompose(cls, mol: Mol) -> Optional[List[SMILES]]:
        try:
            return cls.fragmentation.decompose(mol)
        except Exception:
            return None