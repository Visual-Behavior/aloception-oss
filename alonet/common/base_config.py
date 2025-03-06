import argparse
import yaml
import os
from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class TrainingConfig:
    batch_size: int = 24
    epochs: int = 20
    learning_rate: float = 1e-4
    scheduler_step_size: int = 15
    run_name: str = None
    checkpoint_metric: str = "delta_1"
    load_run: str = None
    load_best: bool = False
    project_path: str = "/home/aloception/.aloception/s2d2/"
    no_compile: bool = False
    overfit: bool = False

    data_path: str = "/data"
    dataset: str = "kitti"
    height: int = 192
    width: int = 640
    sources: tuple[str] = ("prev", "next")
    num_scales: int = 4
    num_workers: int = 8
    no_inv: bool = False
    distribution: str = "gaussian"
    components: int = 2
    alpha_entropy: float = 0.0
    alpha_smooth: float = 0.0
    sigma_entropy: float = 1e-3
    sigma_loss: float = 1.0
    smoothness: float = 0.0
    crop: bool = False
    alpha_loss: str = "ce"  # "ce", "mse", "mae", "attn"
    enc_name: str = "resnet18"
    loss: str = "analytic"  # "analytic", "ssim_mc", "component_mc", "mixture_mc"
    color_depth_grad_w: bool = False
    filter_min: str = "default"  # "default", "global", "alpha"
    car_mask: str = None  # None, "per_pixel", or "per_instance"
    car_mask_weight: float = 0.0
    eigen_old: bool = False
    logger: str = "tensorboard"  # "tensorboard", "mlflow"

    config: Optional[str] = None

    def __post_init__(self):
        # Expand any environment variables in the path strings.
        self.project_path = os.path.expandvars(self.project_path)
        self.data_path = os.path.expandvars(self.data_path)

    @classmethod
    def from_args(cls, args=None) -> "TrainingConfig":
        """Create a TrainingConfig instance from command line arguments and/or config file.

        Returns:
            TrainingConfig: Configuration instance with parsed values
        """
        parser = argparse.ArgumentParser(description="Training configuration")

        # Add argument for config file
        parser.add_argument("--config", type=str, help="Path to config YAML file")

        # Add arguments for all fields in the dataclass, excluding the config field
        for field in fields(cls):
            if field.name != "config":  # Skip config field
                if field.type == bool:
                    parser.add_argument(
                        f"--{field.name}",
                        action="store_true",
                        default=field.default,
                        help=f'{field.name.replace("_", " ").title()}',
                    )
                elif field.name == "sources":
                    # Special handling for sources tuple
                    parser.add_argument(
                        "--sources",
                        nargs="+",  # Accept one or more arguments
                        type=str,
                        default=field.default,
                        help="Source types (e.g., 'prev next' or 'stereo')",
                    )
                else:
                    parser.add_argument(
                        f"--{field.name}",
                        type=field.type,
                        default=field.default,
                        help=f'{field.name.replace("_", " ").title()}',
                    )

        args = parser.parse_args(args)

        # Convert sources list to tuple
        if isinstance(args.sources, list):
            args.sources = tuple(args.sources)

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

    @property
    def HW(self):
        return self.height, self.width

    def to_dict(self):
        return {k: v for k, v in vars(self).items() if v is not None and k != "config"}

    def __str__(self) -> str:
        return str(self.to_dict())

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


if __name__ == "__main__":
    config = TrainingConfig()
    print(config.cls)
