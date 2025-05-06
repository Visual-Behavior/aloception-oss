from argparse import ArgumentParser, Namespace
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import datetime
import os
from collections import OrderedDict
from typing import Optional, Union, Type, TypeVar, Any, Iterator, Tuple
import yaml

parser = ArgumentParser()


def vb_folder(create_if_not_found=False) -> str:
    """
    Get the .aloception folder path

    Parameters
    ----------
    create_if_not_found : bool, optional
        Create the folder if it does not exist, by default False

    Returns
    -------
    str
        The .aloception folder path
    """
    home = os.getenv("HOME")
    alofolder = os.path.join(home, ".aloception")
    if not os.path.exists(alofolder):
        if create_if_not_found:
            print(f"{alofolder} cannot be found so will be created.")
            os.mkdir(alofolder)
        else:
            raise Exception(
                f"{alofolder} do not exist. Please, create the folder with the appropriate files.\
                    (Checkout documentation)"
            )
    return alofolder


def get_expe_infos(project: str, expe_name: str, no_suffix: bool = False) -> Tuple[str, str, str]:
    """
    Get the directories for the project and the experimentation
    A date suffix is added to the expe_name

    Parameters
    ----------
    project : str
        The project name
    expe_name : str
        The experiment name
    no_suffix : bool, optional
        No suffix, by default False

    Returns
    -------
    Tuple[str, str, str]
        The project directory, the experiment directory and the experiment name
    """
    if not no_suffix:
        expe_name = "{}_{:%B-%d-%Y-%Hh-%M}".format(expe_name, datetime.datetime.now())
    project_dir = os.path.join(vb_folder(), f"project_{project}")
    expe_dir = os.path.join(project_dir, expe_name)
    return project_dir, expe_dir, expe_name


def get_expe_infos_from_checkpoint_path(checkpoint_path: str) -> Tuple[str, str, str]:
    """
    Get the directories for the project and the experimentation from the checkpoint path

    Parameters
    ----------
    checkpoint_path : str
        The checkpoint path

    Returns
    -------
    Tuple[str, str, str]
        The project directory, the experiment directory and the experiment name
    """
    expe_dir = os.path.dirname(checkpoint_path)
    expe_name = os.path.basename(expe_dir)
    project_dir = os.path.dirname(expe_dir)
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
    """
    Get the world size of the distributed training

    Returns
    -------
    int
        The world size
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_dist_avail_and_initialized():
    """
    Check if distributed training is available and initialized

    Returns
    -------
    bool
        True if distributed training is available and initialized, False otherwise
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    """
    Get the rank of the current process

    Returns
    -------
    int
        The rank of the current process
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_rank():
    """
    Check if the current process is the main process

    Returns
    -------
    bool
        True if the current process is the main process, False otherwise
    """
    return (is_dist_avail_and_initialized() and get_rank() == 0) or not is_dist_avail_and_initialized()


def only_main_rank(func):
    """
    Decorator to run a function only on the main process
    """

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

    Parameters
    ----------
    dir_path : str
        The directory path

    Returns
    -------
    Optional[str]
        The latest checkpoint path
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

    Parameters
    ----------
    dir_path : str
        The directory path
    condition : str, optional
        The condition, by default "max"

    Returns
    -------
    Optional[str]
        The best checkpoint path
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

    Parameters
    ----------
    step : int
        The step

    Returns
    -------
    str
        The latest checkpoint name
    """
    return f"latest_step={step}"


def topk_cp_name(metric_name: str, metric_value: float, step: int, epoch: int) -> str:
    """
    Get the topk checkpoint name from the step

    Parameters
    ----------
    metric_name : str
        The metric name
    metric_value : float
        The metric value
    step : int
        The step
    epoch : int
        The epoch

    Returns
    -------
    str
        The topk checkpoint name
    """
    return f"epoch={epoch}_step={step}_{metric_name}={metric_value:.5f}"


def get_model_state_dict(model: torch.nn.Module) -> OrderedDict:
    """
    Get the state dict of the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to get the state dict from. This function works with distributed model and
        compiled model.

    Returns
    -------
    OrderedDict
        The state dict of the model.
    """
    if is_dist_avail_and_initialized():
        if is_main_rank():
            model_state_dict = getattr(model.module, "_orig_mod", model.module).state_dict()
    else:
        model_state_dict = getattr(model, "_orig_mod", model).state_dict()
    return model_state_dict


def setup_data_fetcher(dataloader: DataLoader) -> Iterator[Any]:
    """
    Fetch a minibatch from the dataloader

    Parameters
    ----------
    dataloader : DataLoader
        The dataloader to fetch the batch from

    Returns
    -------
    batch : Any
        The batch fetched from the dataloader
    """
    while True:
        for batch in dataloader:
            yield batch
