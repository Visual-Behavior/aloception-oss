from argparse import ArgumentParser, Namespace
import torch
import torch.distributed as dist
import datetime
import os
from collections import OrderedDict
from typing import Optional, Union, Type, TypeVar
import yaml

parser = ArgumentParser()


def vb_folder(create_if_not_found=False):
    home = os.getenv("HOME")
    alofolder = os.path.join(home, ".aloception")
    if not os.path.exists(alofolder):
        if create_if_not_found:
            print(f"{alofolder} cannot be found so will be created.")
            os.mkdir(alofolder)
        else:
            raise Exception(
                f"{alofolder} do not exist. Please, create the folder with the appropriate files. (Checkout documentation)"
            )
    return alofolder


def add_common_training_args(parent_parser: ArgumentParser):
    """add cli argparse arguments to parent_parser

    Parameters
    ----------
    parent_parser: argparse.ArgumentParser
            The custom cli arguments parser, which will be extended by
            the class's default arguments.


    Returns
    -------
    parent_parser: argparse.ArgumentParser
             original parent_parser with added arguments

    Raises
    ------
    RuntimeError:
        If ``parent_parser`` is not an ``ArgumentParser`` instance
    """
    parser = parent_parser.add_argument_group("common_training_argument")
    parser.add_argument("--model_config", type=str, help="Path to the model config file")
    parser.add_argument("--data_config", type=str, help="Path to the data config file")
    parser.add_argument("--training_config", type=str, help="Path to the training config file")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to the checkpoint to resume from")

    return parent_parser


def get_expe_infos(project, expe_name, no_suffix: bool = False):
    """
    Get the directories for the project and the experimentation
    A date suffix is added to the expe_name
    """
    if not no_suffix:
        expe_name = "{}_{:%B-%d-%Y-%Hh-%M}".format(expe_name, datetime.datetime.now())
    project_dir = os.path.join(vb_folder(), f"project_{project}")
    expe_dir = os.path.join(project_dir, expe_name)
    return project_dir, expe_dir, expe_name


def params_update(self, args: Namespace = None, kwargs: dict = {}):
    """Update attributes of one class

    Parameters
    ----------
    self :
        self instance of class
    args : Namespace, optional
        Namespace with arguments to update, by default None
    kwargs : dict, optional
        Dictionary with arguments to update, by default {}

    Notes
    -----
    * Arguments in 'kwargs' is more important than 'args'. For that, they will replace attributes in 'args'.
    * self must have the 'add_argparse_args' staticmethod
    """

    # Get default parameters
    defargs = self.add_argparse_args(ArgumentParser())
    defargs = {act.dest: act.default if not act.required else None for act in defargs._actions if act.dest != "help"}

    if not hasattr(self, "_init_kwargs_config"):
        # Set the init kwargs config
        self._init_kwargs_config = kwargs
    else:
        self._init_kwargs_config.update(kwargs)

    # Priority
    if args is None:
        args = kwargs
    else:
        args = vars(args)
        args.update(kwargs)

    # Attributes update
    for var in defargs:
        self.__dict__[var] = args[var] if var in args else defargs[var]


def _int_or_float_type(x):
    if "." in str(x):
        return float(x)
    return int(x)


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_rank(func):
    def _method(*args, **kwargs):
        if get_rank() == 0:
            return func(*args, **kwargs)
        else:
            return
        return func

    return _method


def get_latest_checkpoint_from_dir(dir_path: str) -> Optional[str]:
    """
    Get the latest checkpoint from the directory.
    The latest checkpoint must have the format `latest_steps-<steps>`
    """
    assert os.path.isdir(dir_path), f"{dir_path} is not a directory"
    checkpoints = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
    for cp in checkpoints:
        if "latest" in cp:
            return os.path.join(dir_path, cp)
    return None


def get_best_checkpoint_from_dir(dir_path: str, condition: str = "max") -> Optional[str]:
    """
    Get the best checkpoint from the directory based on the condition
    The best checkpoint must have the format `epoch=<epoch>_step=<step>_<metric>=<value>`
    """
    assert os.path.isdir(dir_path), f"{dir_path} is not a directory"
    assert condition in ["max", "min"], "Condition must be either `max` or `min`"

    checkpoints = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f)) and "latest" not in f]
    if len(checkpoints) == 0:
        return None
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))
    if condition == "max":
        return os.path.join(dir_path, checkpoints[-1])
    else:
        return os.path.join(dir_path, checkpoints[0])


def latest_cp_name(step: int) -> str:
    """
    Get the latest checkpoint name from the step
    """
    return f"latest_step={step}"


def topk_cp_name(metric_name: str, metric_value: float, step: int, epoch: int) -> str:
    """
    Get the topk checkpoint name from the step
    """
    return f"epoch={epoch}_step={step}_{metric_name}={metric_value}"


def get_model_state_dict(model: torch.nn.Module) -> OrderedDict:
    """
    Get the state dict of the model.

    Args:
        model (torch.nn.Module): The model to get the state dict from. This function works with distributed model and
        compiled model.

    Returns:
        OrderedDict: The state dict of the model.
    """
    if is_dist_avail_and_initialized():
        if is_main_rank():
            model_state_dict = getattr(model.module, "_orig_mod", model.module).state_dict()
    else:
        model_state_dict = getattr(model, "_orig_mod", model).state_dict()
    return model_state_dict


TYPE_CLS = TypeVar("TYPE_CLS")


def init_from_config(cls: Type[TYPE_CLS], config: Union[str, dict]) -> TYPE_CLS:
    """
    Initialize the class from a config file or a dictionary

    Parameters
    ----------
    cls : Type[TYPE_CLS]
        The class to initialize
    config : Union[str, dict]
        The config file or the config dictionary

    Returns
    -------
    The initialized class
    """
    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)
    return cls(**config)
