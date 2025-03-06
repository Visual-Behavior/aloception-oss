import argparse
import yaml
from dataclasses import dataclass, fields, field
from typing import Optional, Union, Literal, get_origin, get_args, Tuple


def parse_tuple(value: str, types: Tuple) -> Tuple:
    """Parse a space-separated string into a tuple of specified types."""
    values = value.split()
    if len(values) != len(types):
        raise argparse.ArgumentTypeError(f"Expected {len(types)} values, got {len(values)}")
    return tuple(type_(val) for type_, val in zip(types, values))


@dataclass
class BaseConfig:
    """Base configuration class.

    Extend this class to create a new configuration class.

    Attributes can be parsed from command line arguments or config file.

    Normal attributes:
    ```python
    attribute_name: type = default_value
    attribute_name: type = field(default=default_value, help="help message")
    ```

    Selectable attributes:
    ```python
    attribute_name: Literal["choice1", "choice2"] = "choice1"
    attribute_name: Literal["choice1", "choice2"] = field(default="choice1",help="help message")
    ```

    Tuple attributes:
    ```python
    attribute_name: Tuple[type1, type2] = (type1_default_value, type2_default_value)
    attribute_name: Tuple[type1, type2] = field(default=(type1_default_value, type2_default_value),help="help message")
    ```

    Args:
        project_name: Name of the project
        experiment_name: Name of the experiment
        val_interval: Validation interval
        log_interval: Logging interval
        save_best_k_cp: Number of best checkpoints to save
        no_suffix: Whether to use no suffix for the checkpoint files
    """

    project_name: str = "default"
    experiment_name: str = "default"
    val_interval: Union[int, float] = 1.0
    log_interval: int = 50
    save_best_k_cp: int = field(default=3, metadata={"help": "Number of best checkpoints to save"})
    no_suffix: bool = field(
        default=False,
        metadata={"help": "Whether to use no suffix for the checkpoint files"},
    )

    batch_size: int = field(default=16, metadata={"help": "Batch size"})
    num_workers: int = field(default=4, metadata={"help": "Number of workers"})

    config: Optional[str] = field(default=None, metadata={"help": "Path to config YAML file"})

    @classmethod
    def from_args(cls, args=None) -> "BaseConfig":
        """Create a BaseConfig instance from command line arguments and/or config file.

        Returns:
            BaseConfig: Configuration instance with parsed values
        """
        parser = argparse.ArgumentParser(description="Configuration")

        # Dynamically add arguments for all fields in the dataclass
        for field in fields(cls):
            help_msg = field.metadata.get("help", "")
            if get_origin(field.type) is Literal:
                choices = get_args(field.type)
                parser.add_argument(
                    f"--{field.name}",
                    type=str,
                    choices=choices,
                    default=field.default,
                    help=help_msg,
                )
            elif field.type is bool:
                # Use store_true for boolean fields
                parser.add_argument(f"--{field.name}", action="store_true", help=help_msg)
            elif get_origin(field.type) is tuple:
                # Use custom parser for tuple fields
                types = get_args(field.type)
                parser.add_argument(
                    f"--{field.name}",
                    type=lambda x: parse_tuple(x, types),
                    nargs="+",
                    default=field.default,
                    help=help_msg,
                )
            else:
                parser.add_argument(
                    f"--{field.name}",
                    type=field.type,
                    default=field.default,
                    help=help_msg,
                )

        args = parser.parse_args(args)

        # If config file is provided, update args with config file values
        if args.config:
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)
                # Update args with config file values, but don't override command line args
                arg_dict = vars(args)
                for key, value in config.items():
                    if arg_dict[key] == parser.get_default(key):
                        arg_dict[key] = value

        return cls(**vars(args))

    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items() if v is not None and k != "config"}

    def save(self, filepath: str) -> None:
        """Save the configuration to a YAML file.

        Args:
            filepath: Path where to save the YAML file
        """
        # Convert dataclass to dictionary, excluding None values and config field
        config_dict = self.to_dict()

        # Write to YAML file
        with open(filepath, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)

    def __str__(self) -> str:
        return str(self.to_dict())
