from fragmentation.fragmentation import Unit, Connection, Fragmentation, FragmentedGraph, BlockLibrary
from rdkit import Chem
from rdkit.Chem import BRICS
from fragmentation.utils import *

class BRICS_Unit(Unit):
    def to_fragment(self, connection):
        assert connection in self.connections
        atom_map = {}
        submol = self.graph.get_submol([self], atom_map=atom_map)
        if self == connection.units[0]:
            atom_index = atom_map[connection.atom_indices[0]]
            brics_label = connection.brics_labels[0]
        else:
            atom_index = atom_map[connection.atom_indices[1]]
            brics_label = connection.brics_labels[0]
        bondtype = connection.bondtype

        rwmol = Chem.RWMol(submol)
        add_dummy_atom(rwmol, atom_index, bondtype, brics_label)
        fragment = rwmol.GetMol()
        fragment = Chem.MolFromSmiles(Chem.MolToSmiles(fragment))
        return fragment
    
class BRICS_Connection(Connection):
    def __init__(self, unit1, unit2, atom_index1, atom_index2, brics_label1: str|int, brics_label2: str|int, bond_index, bondtype):
        super().__init__(unit1, unit2, atom_index1, atom_index2, bond_index, bondtype)
        self.brics_labels = (int(brics_label1), int(brics_label2))

class BRICS_FragmentedGraph(FragmentedGraph):
    def fragmentation(self, mol):
        brics_bond = list(BRICS.FindBRICSBonds(mol))

        rwmol = Chem.RWMol(mol)
        for (atom_idx1, atom_idx2), _ in brics_bond:
            remove_bond(rwmol, atom_idx1, atom_idx2)
        broken_mol = rwmol.GetMol()

        atomMap = Chem.GetMolFrags(broken_mol)
        units = tuple(BRICS_Unit(self, atom_indices) for atom_indices in atomMap)

        unit_map = {}
        for unit in units:
            for idx in unit.atom_indices:
                unit_map[idx] = unit

        connections = []
        for brics_bond in brics_bond:
            (atom_idx1, atom_idx2), (brics_label1, brics_label2) = brics_bond
            bond = mol.GetBondBetweenAtoms(atom_idx1, atom_idx2)
            assert bond is not None
            bond_index, bondtype = bond.GetIdx(), bond.GetBondType()
            unit1 = unit_map[atom_idx1]
            unit2 = unit_map[atom_idx2]
            connection = BRICS_Connection(
                unit1, unit2, atom_idx1, atom_idx2, brics_label1, brics_label2, bond_index, bondtype
            )
        connections = (connection)
        
        return units, connections
    
def brics_fragmentation(mol):
    return BRICS_FragmentedGraph(mol)

class BRICS_Fragmentation(Fragmentation):
    fragmentation = staticmethod(brics_fragmentation)

class BRICS_BlockLibrary(BlockLibrary):
    fragmentation = BRICS_Fragmentation()

    @property
    def brics_label_list(self):
        def get_brics_label(rdmol: Mol):
            return str(rdmol.GetAtomWithIdx(0).GetIsotope())
        
        brics_labels: list[str] = [get_brics_label(rdmol) for rdmol in self.rdmol_list]
        return brics_labels