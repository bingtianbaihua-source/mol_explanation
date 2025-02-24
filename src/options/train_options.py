import argparse
from typing import List, Optional


class TrainArgParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self._setup_argument_groups()

    def _setup_argument_groups(self):
        """Define and organize argument groups."""
        self._add_experiment_args()
        self._add_model_args()
        self._add_data_args()
        self._add_hyperparameter_args()
        self._add_training_config_args()

    def _add_experiment_args(self):
        """Add arguments related to experiment setup."""
        exp_args = self.add_argument_group("Experiment Information")
        exp_args.add_argument("--name", type=str, required=True, help="Job name")
        exp_args.add_argument("--exp_dir", type=str, default="./result/", help="Path of experiment directory")
        exp_args.add_argument("-p", "--property", type=str, nargs="+", help="Property list")

    def _add_model_args(self):
        """Add arguments related to model configuration."""
        model_args = self.add_argument_group("Model Config")
        model_args.add_argument("--model_config", type=str, default="./config/model.yaml", help="Path to model config file")

    def _add_data_args(self):
        """Add arguments related to data configuration."""
        data_args = self.add_argument_group("Data Config")
        data_args.add_argument("--data_dir", type=str, default="./data/ZINC/", help="Dataset directory")

    def _add_hyperparameter_args(self):
        """Add arguments related to training hyperparameters."""
        hparams_args = self.add_argument_group("Training Hyperparameter Config")
        hparams_args.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
        hparams_args.add_argument("--max_step", type=int, default=100000, help="Max training steps (200k used in paper)")
        hparams_args.add_argument("--train_batch_size", type=int, default=512, help="Training batch size")
        hparams_args.add_argument("--val_batch_size", type=int, default=256, help="Validation batch size")
        hparams_args.add_argument("--num_validate", type=int, default=5, help="Number of validation iterations")
        hparams_args.add_argument("--condition_noise", type=float, default=0.02, help="Condition noise")
        hparams_args.add_argument("--num_negative_samples", type=int, default=10, help="Negative sampling hyperparameter")
        hparams_args.add_argument("--alpha", type=float, default=0.75, help="Negative sampling hyperparameter")

        # Loss multipliers
        hparams_args.add_argument("--lambda_term", type=float, default=1.0, help="Termination loss multiplier")
        hparams_args.add_argument("--lambda_property", type=float, default=1.0, help="Property loss multiplier")
        hparams_args.add_argument("--lambda_block", type=float, default=1.0, help="Block loss multiplier")
        hparams_args.add_argument("--lambda_atom", type=float, default=1.0, help="Atom loss multiplier")

    def _add_training_config_args(self):
        """Add arguments related to training configuration."""
        train_args = self.add_argument_group("Training Config")
        train_args.add_argument("--gpus", type=int, default=1, help="Number of GPUs (0 for CPU, 1 for single GPU)")
        train_args.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
        train_args.add_argument("--val_interval", type=int, default=2000, help="Validation interval (steps)")
        train_args.add_argument("--log_interval", type=int, default=100, help="Logging interval (steps)")
        train_args.add_argument("--print_interval", type=int, default=100, help="Printing interval (steps)")
        train_args.add_argument("--save_interval", type=int, default=10000, help="Model checkpoint interval (steps)")