import argparse

class Denovo_Generation_ArgParser(argparse.ArgumentParser):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter

        generator_args = self.add_argument_group('generator')
        generator_args.add_argument(
            '-g', '--generator_config', type=str, help='generator config file'
        )

        overwrite_args = self.add_argument_group('overwrite generator config')
        overwrite_args.add_argument('--model_path', type=str, help='model path')
        overwrite_args.add_argument('--library_path', type=str, help='library path')
        overwrite_args.add_argument('--library_builtin_model_path', type=str, help='builting model path')

        opt_args = self.add_argument_group('optional')
        opt_args.add_argument('-o', '--output_path', type=str)
        opt_args.add_argument('-n', '--num_samples', type=int)
        opt_args.add_argument('--seed', type=int, help='readom seed')
        opt_args.add_argument('-q', action='store_true')

class Scaffold_Generation_ArgParser(Denovo_Generation_ArgParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        scaffold_args = self.add_argument_group('scaffold opt')
        scaffold_args.add_argument('-s', '--scaffold', type=str, default=None, help='scaffold SMILES')
        scaffold_args.add_argument('-S', '--scaffold_path', type=str, default=None, help='scaffold file')
