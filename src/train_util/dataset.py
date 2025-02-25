from torch.utils.data import Dataset
from utils.typing import SMILES
from rdkit import Chem
from rdkit.Chem import Mol
from fragmentation.brics import BRICS_FragmentedGraph, BRICS_BlockLibrary
from torch_geometric.data import Data as PyGData
from torch import FloatTensor, BoolTensor
from transform.core import CoreGraphTransform
from tqdm import tqdm
import torch
import numpy as np


class MyDataset(Dataset):
    def __init__(
        self,
        molecules: list[SMILES | Mol],
        fragmented_molecules: list[BRICS_FragmentedGraph],
        properties: list[dict[str, float]],
        library: BRICS_BlockLibrary,
        library_pygdata_list: list[PyGData],
        library_frequency: FloatTensor,
        num_negative_samples: int,
        train: bool,
    ) -> None:
        super().__init__()
        assert len(molecules) == len(properties), "Molecules and properties must have the same length."

        self.molecules = molecules
        self.properties = properties
        self.library = library
        self.library_pygdata = library_pygdata_list
        self.library_frequency = library_frequency
        self.num_negative_samples = num_negative_samples
        self.train = train
        self.core_transform = CoreGraphTransform.call

        # Fragment molecules if not provided
        if fragmented_molecules is None:
            fragmentation = self.library.fragmentation
            self.fragmented_molecules = [fragmentation(mol) for mol in tqdm(molecules, desc="Fragmenting molecules")]
        else:
            assert len(molecules) == len(fragmented_molecules), "Molecules and fragmented molecules must have the same length."
            self.fragmented_molecules = fragmented_molecules

    def __len__(self) -> int:
        return len(self.fragmented_molecules)

    def __getitem__(self, index: int):
        core_rdmol, block_idx, core_atom_idx = self.get_datapoint(index)
        pygdata_core = self.core_transform(core_rdmol)

        condition = self.properties[index]
        num_core_atoms = core_rdmol.GetNumAtoms()

        # Set target labels
        is_terminal = block_idx is None
        y_term = torch.tensor([is_terminal], dtype=torch.bool)
        y_atom = torch.zeros(num_core_atoms, dtype=torch.bool) if is_terminal else torch.scatter(
            torch.zeros(num_core_atoms, dtype=torch.bool), 0, torch.tensor([core_atom_idx]), torch.tensor([True])
        )

        pygdata_core.y_term = y_term
        pygdata_core.y_atom = y_atom

        # positive and negative samples
        if self.train:
            pos_pygdata = self.library_pygdata[0] if is_terminal else self.library_pygdata[block_idx]
            neg_idxs = [0] * self.num_negative_samples if is_terminal else self.get_negative_samples(block_idx)
            neg_pygdatas = [self.library_pygdata[idx] for idx in neg_idxs]
            return pygdata_core, condition, pos_pygdata, *neg_pygdatas
        else:
            pos_idx = 0 if is_terminal else block_idx
            neg_idxs = [0] * self.num_negative_samples if is_terminal else self.get_negative_samples(block_idx)
            return pygdata_core, condition, pos_idx, *neg_idxs

    def get_datapoint(self, index: int):
        fragmented_mol = self.fragmented_molecules[index]
        min_length, max_length = 2, len(fragmented_mol) + 1
        length_list = np.arange(min_length, max_length + 1)
        length = np.random.choice(length_list, p=length_list / length_list.sum())
        traj = fragmented_mol.get_subtrajectory(length)

        core_rdmol, block_rdmol, (core_atom_idx, _) = fragmented_mol.get_datapoint(traj)
        block_idx = self.library.get_index(block_rdmol) if block_rdmol is not None else None
        return core_rdmol, block_idx, core_atom_idx

    def get_negative_samples(self, positive_sample: int) -> list[int]:
        freq = self.library_frequency.clone()
        freq[positive_sample] = 0.0  # Exclude the positive sample
        return torch.multinomial(freq, self.num_negative_samples, replacement=True).tolist()