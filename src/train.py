from model.layer.block.graph_conv import ResidualBlock
from model.layer.block.readout import Readout
from fragmentation.utils import *
from utils.typing import SMILES
from fragmentation.brics import BRICS_FragmentedGraph
from fragmentation.brics import BRICS_BlockLibrary
from transform.core import CoreGraphTransform
from torch_geometric.data import Data as PyGData, Batch as PyGBatch