import logging
import os
from torch.utils.tensorboard.writer import SummaryWriter
from fragmentation.brics import BRICS_BlockLibrary
from transform.core import CoreGraphTransform
from torch_geometric.data import Data as PyGData, Batch as PyGBatch
import pandas as pd
import pickle


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
            molecules = []

