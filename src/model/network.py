import torch
from torch import Tensor, LongTensor
import torch.nn as nn
from typing import OrderedDict
from model.layer.graph_embedding import GraphEmbeddingModel
from model.layer.property_prediction import MultiHeadPropertyClassificationModel
from model.layer.condition_embedding import ConditionEmbeddingModel
from model.layer.termination_prediction import TerminationPredictionModel
from model.layer.atom_selection import AtomSelectionModel
from model.layer.block_selection import BlockSelectionModel
from utils.typing import *
from transform import NUM_ATOM_FEATURES, NUM_BLOCK_FEATURES, NUM_BOND_FEATURES
from torch_geometric.data import Data as PyGData, Batch as PyGBatch
from torch_geometric.typing import Adj

class BlockConnectionPredictor(nn.Module):

    def __init__(self, 
                 config,
                 property_information: OrderedDict[str, tuple[float, float]]):
        super(BlockConnectionPredictor, self).__init__()
        self.config = config
        self.property_information = property_information
        self.property_dim = len(property_information)

        self.core_graph_embedding = GraphEmbeddingModel(
            NUM_ATOM_FEATURES, NUM_BOND_FEATURES, 0, **config.GraphEmbeddingModel_core)

        self.block_graph_embedding = GraphEmbeddingModel(
            NUM_ATOM_FEATURES, NUM_BOND_FEATURES, NUM_BLOCK_FEATURES, **config.GraphEmbeddingModel_core)
        
        num_classes_list = [2 for _ in range(self.property_dim)]
        self.property_prediction = MultiHeadPropertyClassificationModel(
            num_classes=num_classes_list, **config.PropertyPredictionModel)

        self.condition_embedding = ConditionEmbeddingModel(
            condition_dim=self.property_dim, **config.ConditionEmbedding)
        
        self.termination_prediction = TerminationPredictionModel(
            **config.TerminationPrediction)

        self.block_selection = BlockSelectionModel(
            **config.BlockSelectionModel)

        self.atom_selection = AtomSelectionModel(
            NUM_BOND_FEATURES, **config.AtomSelection)

    def core_molecule_embedding(self, batch: PyGBatch | PyGData):
        return self.core_graph_embedding.forward_batch(batch)
    
    def building_block_embedding(self, batch: PyGBatch | PyGData):
        return self.block_graph_embedding.forward_batch(batch)
    
    def get_property_prediction(self, Z_core: GlobalVector):
        return self.property_prediction(Z_core)
    
    def embed_condition(self, x_upd_core: NodeVector, Z_core: GraphVector, condition: dict[str, int], node2graph_core: LongTensor = None):
        return self.condition_embedding(x_upd_core, Z_core, condition, node2graph_core)
    
    def get_termination_logit(self, Z_core: GraphVector):
        return self.termination_prediction(Z_core, return_logit=True)
        
    def get_termination_probability(self, Z_core: GraphVector):
        return self.termination_prediction(Z_core, return_logit=False)
    
    def get_block_priority(self, Z_core: GraphVector, Z_block: GraphVector):
        return self.block_selection(Z_core, Z_block)
    
    def get_atom_probability_distribution(self, x_upd_core: NodeVector, edge_index_core: Adj, edge_attr_core: EdgeVector, Z_core: GraphVector, Z_block: GraphVector, node2graph_core: LongTensor = None):
        return self.atom_selection(x_upd_core, edge_index_core, edge_attr_core, Z_core, Z_block, node2graph_core)
    
    def get_atom_probability_logit(self, x_upd_core: NodeVector, edge_index_core: Adj, edge_attr_core: EdgeVector, Z_core: GraphVector, Z_block: GraphVector, node2graph_core: LongTensor = None):
        return self.atom_selection(x_upd_core, edge_index_core, edge_attr_core, Z_core, Z_block, node2graph_core, return_logit=True)
    
    def initialize_model(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    def save(self, save_path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'property_information': self.property_information
        }, save_path)
        
    @classmethod
    def load_from_file(cls, checkpoint_path: str, map_location: str = 'cpu'):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        return cls.load_from_checkpoint(checkpoint, map_location)
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint: dict, map_location: str = 'cpu'):
        model = cls(checkpoint['config'], checkpoint['property_information'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(map_location)
        return model