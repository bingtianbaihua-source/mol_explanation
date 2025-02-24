from torch.utils.data import Dataset
from 

class MyDataset(Dataset):
    def __init__(self,
                 mols: list[]) -> None:
        super().__init__()