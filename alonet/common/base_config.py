import argparse
from argparse import Namespace, ArgumentParser, BooleanOptionalAction
import yaml
from dataclasses import dataclass, fields, field, Field
from typing import Optional, get_origin, get_args, Literal


def add_argument(parser: argparse.ArgumentParser, field_args: Field, prefix: str = None) -> None:
    """
    Add an argument to the argument parser

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser
    field_args : Field
        Field to add to the argument parser
    prefix : str
        Prefix for the argument
    """
    help_msg = field_args.metadata.get("help", "")
    help_msg = help_msg + f". Default: {field_args.default}"
    arg_name = field_args.name if prefix is None else f"{prefix}.{field_args.name}"

    if field_args.type is bool:
        # Use store_true for boolean fields
        help_msg = help_msg + f". Use --no-{arg_name} to get False as value."
        parser.add_argument(f"--{arg_name}", action=BooleanOptionalAction, help=help_msg, default=field_args.default)
    elif get_origin(field_args.type) is tuple:
        # Use custom parser for tuple fields
        parser.add_argument(
            f"--{arg_name}",
            nargs="+",
            default=field_args.default,
            help=help_msg,
        )
    elif get_origin(field_args.type) is Literal:
        # Add choices
        parser.add_argument(
            f"--{arg_name}",
            choices=get_args(field_args.type),
            default=field_args.default,
            help=help_msg,
        )
    else:
        parser.add_argument(
            f"--{arg_name}",
            type=field_args.type,
            default=field_args.default,
            help=help_msg,
        )


def nested_to_flat_dict(nested_dict: dict, prefix: Optional[str] = None) -> dict:
    """
    Convert a nested dictionary to a flat dictionary.
    """
    flat_dict = {}
    for key, value in nested_dict.items():
        if prefix is not None:
            key = f"{prefix}.{key}"
        else:
            key = f"{key}"
        if isinstance(value, dict):
            flat_dict.update(nested_to_flat_dict(value, prefix=key))
        else:
            flat_dict[key] = value
    return flat_dict


@dataclass
class BaseConfig:
    """Base configuration class for composite configurations.

    This class extends BaseConfig and adds support for nested configurations.
    It allows for the composition of multiple configurations into a single configuration.

    Attributes:
        config_file: Path to the configuration file
        Other attributes are classes derived from BaseConfig.

    Methods:
        add_argparse_args: Add arguments to the argument parser
        from_args: Create a BaseConfig instance from command line arguments and/or config file
        to_dict: Convert the BaseConfig instance to a dictionary
        save: Save the BaseConfig instance to a file
        __str__: Convert the BaseConfig instance to a string
    """

    @classmethod
    def add_argparse_args(cls, parent_parser: Optional[ArgumentParser] = None, prefix: str = None) -> ArgumentParser:
        """
        Add arguments to the argument parser

        Parameters
        ----------
        parent_parser : Optional[ArgumentParser]
            Parent argument parser
        prefix : str
            Prefix for the arguments

        Returns
        -------
        ArgumentParser
            Argument parser
        """
        if parent_parser is None:
            parent_parser = ArgumentParser(description="Configuration")
        # Dynamically add arguments for all fields in the dataclass
        for field_args in fields(cls):
            if hasattr(field_args.type, "add_argparse_args"):
                arg_name = field_args.name if prefix is None else f"{prefix}.{field_args.name}"
                field_args.type.add_argparse_args(parent_parser, prefix=arg_name)
            else:
                add_argument(parent_parser, field_args, prefix=prefix)

        config_file_arg = "--config_file" if prefix is None else f"--{prefix}.config_file"
        parent_parser.add_argument(config_file_arg, type=str, default=None, help="Path to the configuration file")

        return parent_parser

    @classmethod
    def from_args(cls, args: Namespace) -> "BaseConfig":
        """
        Create a BaseConfig instance from command line arguments and/or config file.

        Parameters
        ----------
        args : Namespace
            Command line arguments
        """
        arg_dict = vars(args)
        parser = ArgumentParser()
        cls.add_argparse_args(parser)

        # If config file is provided, update args with config file values
        config = {}
        if args.config_file:
            with open(args.config_file, "r") as f:
                config = yaml.safe_load(f)
                config = nested_to_flat_dict(config)

        for arg_key in arg_dict:
            if arg_key in config and arg_dict[arg_key] == parser.get_default(arg_key):
                arg_dict[arg_key] = config[arg_key]

        # Initialize the dataclass with the parsed arguments
        parameters = {}
        for field_args in fields(cls):
            if hasattr(field_args.type, "from_args"):
                sub_args = {}
                for arg_key in arg_dict:
                    # Check if the prefix exists in the arg_key
                    if field_args.name not in arg_key:
                        continue
                    keywords = arg_key.split(".")
                    key, sub_key = keywords[0], ".".join(keywords[1:])
                    if key == field_args.name:
                        # If the value is defined in the config file and the default value is the same as the command
                        # line arg, use the config file value. Else, do not override the command line arg.
                        if arg_key in config and arg_dict[arg_key] == parser.get_default(arg_key):
                            sub_args[sub_key] = config[arg_key]
                        else:
                            sub_args[sub_key] = arg_dict[arg_key]
                sub_args = Namespace(**sub_args)
                parameters[field_args.name] = field_args.type.from_args(sub_args)
            else:
                # If the value is defined in the config file and the default value is the same as the command
                # line arg, use the config file value. Else, do not override the command line arg.
                if field_args.name in config and arg_dict[field_args.name] == parser.get_default(field_args.name):
                    parameters[field_args.name] = config[field_args.name]
                else:
                    parameters[field_args.name] = getattr(args, field_args.name)

        return cls(**parameters)

    def to_dict(self) -> dict:
        """
        Convert the BaseConfig instance to a dictionary.

        Returns
        -------
        dict
        """
        data = {}
        for field_args in fields(self):
            if hasattr(field_args.type, "to_dict"):
                data[field_args.name] = getattr(self, field_args.name).to_dict()
            else:
                data[field_args.name] = getattr(self, field_args.name)
        return data

    def save(self, filepath: str) -> None:
        """
        Save the BaseConfig instance to a file.

        Parameters
        ----------
        filepath : str
            Filepath to save the BaseConfig instance
        """
        data = self.to_dict()
        with open(filepath, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    def __str__(self) -> str:
        return str(self.to_dict())


@dataclass
class BaseTrainerConfig(BaseConfig):
    """Base trainer configuration class."""

    # BaseTrainer
    project_name: str = field(default="default", metadata={"help": "Name of the project"})
    experiment_name: str = field(default="default", metadata={"help": "Name of the experiment"})
    accumulate_grad_batches: int = field(default=1, metadata={"help": "Number of batches to accumulate gradients"})
    num_epochs: int = field(default=None, metadata={"help": "Number of epochs to train"})
    num_steps: int = field(default=None, metadata={"help": "Number of steps to train"})
    val_interval: float = field(default=1.0, metadata={"help": "Validation interval"})
    val_every_n_steps: int = field(default=None, metadata={"help": "Validation interval in steps"})
    log_interval: int = field(default=50, metadata={"help": "Logging interval"})
    save_best_k_cp: int = field(default=3, metadata={"help": "Number of best checkpoints to save"})
    logger: str = field(default=None, metadata={"help": "Logger to use"})
    resume: bool = field(default=False, metadata={"help": "Whether to resume training"})
    no_suffix: bool = field(default=False, metadata={"help": "Whether to use no suffix for the checkpoint files"})
    checkpoint: str = field(default=None, metadata={"help": "Path to checkpoint file"})


@dataclass
class BaseDataModuleConfig(BaseConfig):
    """Base data module configuration class."""

    batch_size: int = field(default=16, metadata={"help": "Batch size"})
    num_workers: int = field(default=4, metadata={"help": "Number of workers"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = BaseTrainerConfig.add_argparse_args(parser)
    args = parser.parse_args()
    config = BaseTrainerConfig.from_args(args)
    print(config)
