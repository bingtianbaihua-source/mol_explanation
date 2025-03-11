from fragmentation.brics import BRICS_BlockLibrary, BRICS_Fragmentation

import os
from pathlib import Path
import logging
from omegaconf import OmegaConf
import torch
from train_util.trainer import Trainer
from options.train_options import TrainArgParser
from logger.logger import setup_logger
from utils.seed import set_seed

import warnings

warnings.filterwarnings('ignore')
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    set_seed(0)
    parser = TrainArgParser()
    args = parser.parse_args()

    run_dir = os.path.join(args.exp_dir, args.name)
    assert not os.path.exists(run_dir), f'{run_dir} is already exsits'
    Path(run_dir).mkdir(parents=True, exist_ok=False)

    setup_logger(run_dir)
    config_path = os.path.join(run_dir, 'config.yaml')
    OmegaConf.save(vars(args), config_path)

    logging.info(
        'Training Information\n'
        + 'Argument\n'
        + '\n'.join([f' {arg}: {getattr(args, arg)}' for arg in vars(args)])
        + '\n'
    )

    trainer = Trainer(args, run_dir)
    trainer.fit()

if __name__ == '__main__':
    main()