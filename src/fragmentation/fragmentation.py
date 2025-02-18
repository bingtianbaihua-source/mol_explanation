from rdkit import Chem
from fragmentation.utils import *


class Unit:
    def __init__(self,
                 graph,
                 atom_indices: tuple[int]):
        self.graph = graph
        atom_indices_set = set(atom_indices)
        bond_indices = []
        for bond in graph.rdmol.GetBonds():
            atom_idx1, atom_idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtoms()
            if atom_idx1 in atom_indices_set & atom_idx2 in atom_indices_set:
                bond_indices.append(bond.GetIdx())

        self.atom_indices = atom_indices
        self.bond_indeces = tuple(bond_indices)

        self.neighbors = []
        self.connections = []

    def add_connection(self, neighbor_unit, connection):
        self.neighbors.append(neighbor_unit)
        self.connections.append(connection)

    def to_rdmol(self):
        return self.graph.get_submol([self])
    
    def to_fragment(self, connection):
        assert connection in self.connections
        atomMap = {}
        submol = self.graph.get_submol([self], atomMap=atomMap)
        if self == connection.units[0]:
            atom_index = atomMap[connection.atom_indices[0]]
        else:
            atom_index = atomMap[connection.atom_indices[1]]
        bondtype = connection.bondtype

        rwmol = Chem.RWMol(submol)
        add_dummy_atom(rwmol, atom_index, bondtype)
        fragment = rwmol.GetMol()
        fragment = Chem.MolFromSmiles(Chem.MolToSmiles(fragment))
        return fragment
    
class Connection:
    def __init__(self) -> None:
        pass
        

