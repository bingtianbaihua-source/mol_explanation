from rdkit import Chem
from rdkit.Chem import Mol, Atom, BondType

def create_bond(rwmol: Chem.RWMol, idx1: int, idx2: int, bondtype: BondType) -> None:
    """在分子中添加键并调整氢原子数量。"""
    rwmol.AddBond(idx1, idx2, bondtype)
    for idx in [idx1, idx2]:
        atom = rwmol.GetAtomWithIdx(idx)
        atom_numexplicitHs = atom.GetNumExplicitHs()
        if atom_numexplicitHs:
            atom.SetNumExplicitHs(atom_numexplicitHs - 1)

def remove_bond(rwmol: Chem.RWMol, idx1: int, idx2: int) -> None:
    """在分子中删除键并调整氢原子数量。"""
    rwmol.RemoveBond(idx1, idx2)
    for idx in [idx1, idx2]:
        atom = rwmol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'N' and atom.GetIsAromatic():
            atom.SetNumExplicitHs(1)

def check_dummy_atom(atom: Chem.Atom) -> bool:
    """检查原子是否为虚拟原子。"""
    return atom.GetAtomicNum() == 0

def add_dummy_atom(rwmol: Chem.RWMol, index: int, bondtype: BondType = BondType.SINGLE, label: int = 0) -> None:
    """在分子中添加虚拟原子并创建键。"""
    dummy_atom = Atom('*')
    dummy_atom.SetIsotope(label)
    new_idx = rwmol.AddAtom(dummy_atom)
    create_bond(rwmol, index, new_idx, bondtype)

def find_dummy_atom(rdmol: Mol) -> int:
    """查找分子中的虚拟原子并返回其索引。"""
    for idx, atom in enumerate(rdmol.GetAtoms()):
        if check_dummy_atom(atom):
            return idx
    return -1  # 使用-1表示未找到虚拟原子

def get_dummy_bondtype(dummy_atom: Chem.Atom) -> BondType:
    """获取虚拟原子的键类型。"""
    bondtype = dummy_atom.GetTotalValence()
    if bondtype == 1:
        return BondType.SINGLE
    elif bondtype == 2:
        return BondType.DOUBLE
    elif bondtype == 3:
        return BondType.TRIPLE
    else:
        raise ValueError("Invalid bond type for dummy atom")

def create_monoatomic_mol(smi: str) -> Mol:
    """从SMILES字符串创建单原子分子。"""
    return Chem.MolFromSmiles(smi)

def merge(scaffold: Mol, fragment: Mol, index1: int, index2: int = None) -> Mol:
    """合并两个分子片段。"""
    if index2 is None:
        index2 = find_dummy_atom(fragment)
        if index2 == -1:
            raise ValueError("No dummy atom found in fragment")

    rwmol = Chem.RWMol(Chem.CombineMols(scaffold, fragment))
    dummy_atom_index = scaffold.GetNumAtoms() + index2

    dummy_atom = rwmol.GetAtomWithIdx(dummy_atom_index)
    if not check_dummy_atom(dummy_atom):
        raise ValueError("Expected a dummy atom at the specified index")
    
    bondtype = get_dummy_bondtype(dummy_atom)
    index2 = dummy_atom.GetNeighbors()[0].GetIdx()
    create_bond(rwmol, index1, index2, bondtype)
    rwmol.RemoveAtom(dummy_atom_index)
    mol = rwmol.GetMol()
    Chem.SanitizeMol(mol)

    return mol