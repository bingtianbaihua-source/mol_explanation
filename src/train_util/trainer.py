import logging
import os
from torch.utils.tensorboard.writer import SummaryWriter
from fragmentation.brics import BRICS_BlockLibrary
from transform.core import CoreGraphTransform
from torch_geometric.data import Data as PyGData, Batch as PyGBatch
from torch_geometric.loader import DataLoader as PyGDataloader
import pandas as pd
import pickle
from train_util.dataset import MyDataset
from collections import OrderedDict
from omegaconf import OmegaConf
from model.network import BlockConnectionPredictor
import torch
from torch import FloatTensor, BoolTensor
from utils.typing import *
from torch import functional as F




class Trainer:
    def __init__(self, args, run_dir: str):
        self.args = args
        
    def setup_trainer(self, args):
        logging.info('Setup Trainer')
        self.device = 'cuda:0' if args.gpus > 0 else 'cpu'
        self.lr = args.lr
        self.max_step = args.max_step
        self.num_negative_samples = args.num_negative_samples
        self.num_validate = args.num_validate
        self.alpha = args.alpha
        self.condition_noise = args.condition_noise

        self.lambda_term = args.lambda_term
        self.lambda_property = args.lambda_property
        self.lambda_block = args.lambda_block
        self.lambda_atom = args.lambda_atom

        self.num_workers = args.num_workers
        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.val_batch_size

        self.val_interval = args.val_interval
        self.save_interval = args.save_interval
        self.print_interval = args.print_interval
        self.log_interval = args.log_interval

    def setup_work_directory(self, run_dir):
        logging.info('Setup work Directory')
        self.save_dir = save_dir = os.path.join(run_dir, 'save')
        self.log_dir = log_dir = os.path.join(run_dir, 'log')
        self.model_config_path = os.path.join(run_dir, 'model_config.yaml')
        os.mkdir(save_dir)
        os.mkdir(log_dir)
        self.tb_logger = SummaryWriter(log_dir=log_dir)

    def setup_library(self, args):
        logging.info('setup library')
        library_path = os.path.join(args.data_dir, 'library.csv')
        self.library = library = BRICS_BlockLibrary(library_path, use_frequency=True)
        self.library_frequency = self.library.frequency_distribution ** args.alpha
        self.library_pygdata_list = [CoreGraphTransform.call(mol) for mol in library]
        self.library_pygbatch = PyGBatch.from_data_list(self.library_pygdata_list)

    def setup_data(self, args):
        logging.info('Setup Data')
        data_path = os.path.join(args.data_dir, 'data.csv')
        dataframe = pd.read_csv(data_path, usecols=['SMILES'] + args.property)
        data_pkl_path = os.path.join(args.data_dir, 'data.pkl')
        if os.path.exists(data_pkl_path):
            with open(data_pkl_path, 'rb') as f:
                data_pkl = pickle.load(open(data_pkl_path, 'rb'))
        else:
            data_pkl = None

        split_path = os.path.join(args.data_dir, 'split.csv')
        with open(split_path) as f:
            split_path = [l.strip().split(',') for l in f.readlines()]
        train_index = [int(idx) for label,idx in split_path if label == 'train']
        val_index = [int(idx) for label,idx in split_path if label == 'val']

        def construct_dataset(dataframe, data_pkl, index, train: bool):
            indexed_df = dataframe.loc[index]
            molecules = [row.SMILES for row in indexed_df.itertuples(index=False)]
            properties = [{key: getattr(row, key) for key in args.property} for row in indexed_df.itertuples(index=False)]
            fragmented_molecules = [data_pkl[idx] for idx in index] if data_pkl is not None else None
            return MyDataset(
                molecules,
                fragmented_molecules,
                properties,
                self.library,
                self.library_pygdata_list,
                self.library_frequency,
                self.num_negative_samples,
                train
            )
        
        self.train_dataset = construct_dataset(dataframe, data_pkl, train_index, train=True)
        self.val_dataset = construct_dataset(dataframe, data_pkl, val_index, train=False)
        self.train_dataloader = PyGDataloader(
            self.train_dataset,
            args.train_batch_size,
            num_workers = args.num_workers,
            shuffle=True,
            drop_last=True
        )

        self.val_dataloader = PyGDataloader(
            self.val_dataset,
            args.val_batch_size,
            num_workers = args.num_workers,
            shuffle=True,
            drop_last=True
        )

        self.property_mean_std = OrderedDict()
        for desc in args.property:
            mean,std = dataframe[desc].mean(), dataframe[desc].std()
            self.property_mean_std[desc] = (mean, std)

        del [[dataframe]]
        
        logging.info(f'num of train data: {len(self.train_dataset)}')
        logging.info(f'num of val data: {len(self.val_dataset)}')

    def setup_model(self, args):
        logging.info('Setup Model')
        model_config = OmegaConf.load(args.model_config)
        OmegaConf.resolve(model_config)
        OmegaConf.save(model_config, self.model_config_path)
        model = BlockConnectionPredictor(model_config, self.property_mean_std)
        model.initialize_model()
        self.model = model.to(self.device)
        logging.info(f'number of parameter: {sum((p.numel) for p in self.model.parameters())}')

    def fit(self):
        optimizer, schedular = self.setup_optimizer()
        self.global_step = 0
        self.global_epoch = 0
        self.min_loss = float('inf')
        self.model.train()
        logging.info('Train start')
        optimizer.zero_grad()
        while self.global_step < self.max_step:
            self.global_epoch += 1
            for batch in self.train_dataloader:
                pass

    def run_train_step(self):
        pass

    def _step(self, batch, train: bool):
        batch = self.to_device(batch)
        pygbatch_core, condition, pos_block, neg_blocks = batch
        y_term, y_atom = pygbatch_core['y_term'], pygbatch_core['y_atom']
        y_add = torch.logical_not(y_term)

        metrics: dict[str, FloatTensor] = {}
        cond = self.standardize_condition(condition)

        x_upd_core, Z_core = self.core_mol_embedding(pygbatch_core)

        if y_term.sum().item() > 0 and self.lambda_property > 0:
            loss_property = self.alpha

    def property_prediction(self, Z_core: GraphVector, cond: PropertyVector, y_term: BoolTensor):
        y_property = cond[y_term]
        Z_core = Z_core[y_term]
        y_hat_property = self.model.get_property_prediction(Z_core)
        loss_property = F.

    def core_mol_embedding(self, pygbatch_core: PyGBatch):
        x_upd_core, Z_core = self.model.molecule_embedding(pygbatch_core, is_core=True)
        return x_upd_core, Z_core

    def standardize_condition(self, condition: dict[str, FloatTensor]):
        cond = self.model.standardize_condition(condition)
        return cond
        

    def to_device(self, batch):
        pygbatch_core, condition, pos_block, *neg_blocks = batch
        pygbatch_core = pygbatch_core.to(self.device)
        condition = {key: val.to(self.device) for key, val in condition.items()}
        pos_block = pos_block.to(self.device)
        neg_blocks = [block.to(self.device) for block in neg_blocks]
        return pygbatch_core, condition, pos_block, neg_blocks


    def setup_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-5)
        return optimizer, schedular


