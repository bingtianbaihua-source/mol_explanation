from rdkit.Chem import BRICS
from rdkit import Chem

smi = 'COc1cc(OC)c(NC(=O)C(=O)Nc2c(OC)cc(OC)cc2OC)c(OC)c1'
mol = Chem.MolFromSmiles(smi)
brics_bond = list(BRICS.FindBRICSBonds(mol))
print(brics_bond)