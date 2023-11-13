from argparse import ArgumentParser, ArgumentTypeError, Namespace, _ArgumentGroup
from typing import Any
from pytorch_lightning import Trainer
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import datetime
import os
import json
import re
import torch

from alodataset.base_dataset import _user_prompt

parser = ArgumentParser()


class AloceptionCLI(LightningCLI):
    def __init__(self, project: str, **kwargs):
        """AloceptionCLI - CLI tool of aloception-oss which wraps LightningCLI.
        Args:
            project (str): Name of project. Necessary to log/load checkpoint/training log.
            compile_model (bool): Compile model via `torch.compile` before training. Default: True.
        """
        self.project = project
        super().__init__(**kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument(
            "--resume",
            action="store_true",
            help="Resume all the training, not only the weights.",
        )
        parser.add_argument(
            "--save",
            action="store_true",
            help="Save every epoch and keep the top_k",
        )
        parser.add_argument(
            "--save_top_k",
            type=int,
            default=3,
            help="Stores up to top_k models, by default %(default)s. Use save_top_k=-1 to store all checkpoints",
        )
        parser.add_argument(
            "--checkpoint",
            type=checkpoint_type,
            default="last",
            help="Load the weights from 'best', 'last' or '.ckpt' file into run_id, by default '%(default)s'",
        )
        parser.add_argument(
            "--no_compile",
            action="store_true",
            help="Do not compile model via torch.compile.",
        )
        parser.add_argument(
            "--log",
            type=str,
            default=None,
            nargs="?",
            const="wandb",
            help="Log results, can specify logger, by default %(default)s. If set but value not provided:wandb",
        )
        parser.add_argument(
            "--log_save_dir",
            type=str,
            default=None,
            nargs="?",
            help="Path to save training log, by default %(default)s. If not set, use the default value in alonet_config.json",
        )
        parser.add_argument(
            "--cp_save_dir",
            type=str,
            default=None,
            nargs="?",
            help="Path to save training checkpoint, by default %(default)s. If not set, use the default value in alonet_config.json",
        )
        parser.add_argument("--run_id", type=str, help="Load the weights from this saved experiment")
        parser.add_argument(
            "--monitor",
            type=str,
            default="val_loss",
            help="Metric to save/load weights, by default '%(default)s'",
        )
        parser.add_argument(
            "--no_run_id",
            action="store_true",
            help="Skip loading form run_id when an experiment is restored.",
        )
        parser.add_argument("--project_run_id", type=str, help="Project related with the run ID to load")
        parser.add_argument(
            "--expe_name",
            type=str,
            default=None,
            help="expe_name to be logged in wandb",
        )
        parser.add_argument(
            "--no_suffix",
            action="store_true",
            help="do not add date suffix to expe_name",
        )
        parser.add_argument(
            "--nostrict",
            action="store_true",
            help="load from checkpoint to run a model with different weights names (default False)",
        )

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        # get callbacks from self.model
        callback_fn = getattr(self.model, "callbacks", None)
        if callable(callback_fn):
            callbacks = callback_fn(self.datamodule)
        else:
            callbacks = []

        extra_callbacks = [self._get(self.config_init, c) for c in self._parser(self.subcommand).callback_keys]
        trainer_config = {**self._get(self.config_init, "trainer", default={}), **kwargs}

        # add extra_callbacks
        callbacks.extend(extra_callbacks)

        # get config from subcommand (fit, evaluate, predict)
        config_subcommand = self.config[self.subcommand]

        project_dir, expe_dir, expe_name = get_expe_infos(self.project, config_subcommand.expe_name, config_subcommand)
        if config_subcommand.run_id is not None:
            strict = not config_subcommand.nostrict
            run_id_project_dir = (
                project_dir
                if config_subcommand.project_run_id is None
                else os.path.join(vb_folder(), f"project_{config_subcommand.project_run_id}")
            )
            ckpt_path = os.path.join(run_id_project_dir, config_subcommand.run_id, "last.ckpt")
            if not os.path.exists(ckpt_path):
                raise Exception(f"Impossible to load the ckpt at the following destination:{ckpt_path}")
            if not config_subcommand.resume:
                kwargs_config = getattr(self.model, "_init_kwargs_config", {})
                print(f"Loading ckpt from {config_subcommand.run_id} at {ckpt_path}")
                self.model = type(self.model).load_from_checkpoint(ckpt_path, strict=strict, **kwargs_config)
            else:
                expe_name = config_subcommand.run_id
                # update ckpt_path in self.config_init for Trainer.fit/validate/test
                self.config_init["fit"]["ckpt_path"] = ckpt_path
                expe_dir = os.path.join(run_id_project_dir, config_subcommand.run_id)

        if config_subcommand.log is not None:
            # get dir to save log
            save_dir = os.path.join(
                get_dir_from_config(config_subcommand, "log"), f"project_{self.project}", expe_name
            )

            # create path
            if config_subcommand.log == "wandb":
                save_dir = os.path.join(save_dir, "wandb")
                os.makedirs(save_dir, exist_ok=True)
                logger = WandbLogger(name=expe_name, save_dir=save_dir, project=self.project, id=expe_name)

            elif config_subcommand.log == "tensorboard":
                save_dir = os.path.join(save_dir, "tensorboard")
                os.makedirs(save_dir, exist_ok=True)
                logger = TensorBoardLogger(save_dir=save_dir, name=expe_name, sub_dir=expe_name)
            else:
                raise ValueError("Unknown or not implemented logger")
        else:
            logger = None

        if config_subcommand.save:
            monitor = getattr(config_subcommand, "monitor", "val_loss")
            checkpoint_callback = ModelCheckpoint(
                dirpath=expe_dir,
                verbose=True,
                save_last=True,
                save_top_k=getattr(config_subcommand, "save_top_k", 3),
                monitor=monitor,
                filename="{epoch}-{step}-{" + monitor + ":.4f}",
            )
            callbacks.append(checkpoint_callback)

        trainer_config.update({"logger": logger})

        if not config_subcommand.no_compile:
            self.model = torch.compile(self.model)
            print("[INFO] Model is compiled succsesfully !")
            # only compile one time
            config_subcommand.no_compile = True

        return self._instantiate_trainer(trainer_config, callbacks)


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


def checkpoint_type(arg_value):
    if arg_value is None:
        return "last"
    elif arg_value.lower() == "last" or arg_value.lower() == "best":
        return arg_value.lower()
    elif os.path.splitext(arg_value)[-1].lower() == ".ckpt":
        return arg_value
    else:
        raise ArgumentTypeError(f"{arg_value} is not a valid checkpoint. Use 'last','best' or '.ckpt' file")


def checkpoint_handler(checkpoint_path, rfolder, monitor="val_loss"):
    if checkpoint_path == "last":
        return "last.ckpt"
    elif os.path.splitext(checkpoint_path)[-1].lower():
        return checkpoint_path
    elif checkpoint_path.lower() == "best":
        best_path, best_monitor, best_epoch = None, None, None
        # First way: load the best model from file name
        for fpath in os.listdir(rfolder):
            try:
                ck_props = dict(
                    map(
                        lambda y: y.split("="),
                        re.findall("[-]*(.*?)(?:-|.ckpt)", fpath),
                    )
                )
            except:  # No information in filename, skipping
                continue
            if monitor not in ck_props:
                continue
            cmonitor, cepoch = float(ck_props[monitor]), int(ck_props["epoch"])
            if best_monitor is None or cmonitor < best_monitor:
                best_path, best_monitor, best_epoch = fpath, cmonitor, cepoch
            elif cmonitor == best_monitor and cepoch < best_epoch:
                best_path, best_monitor, best_epoch = fpath, cmonitor, cepoch

        if best_path is not None:
            print(f"Found best model at {best_path}.")
            return best_path

        # Second way: load the best model using monitor saved in checkpoints
        for fpath in os.listdir(rfolder):
            stact_dict = torch.load(os.path.join(rfolder, fpath), map_location="cpu")
            ck_props = [v for k, v in stact_dict["callbacks"].items() if ModelCheckpoint.__name__ == k.__name__]
            if len(ck_props) == 0 or monitor != ck_props[0]["monitor"]:
                continue
            cmonitor = ck_props[0]["current_score"].item()
            if best_monitor is None or cmonitor < best_monitor:
                best_path, best_monitor = fpath, cmonitor

        if best_path is not None:
            print(f"Found best model at {best_path}.")
            return best_path
        else:
            raise RuntimeError(
                f"Not '{monitor}' found on checkpoints. Use '--checkpoint last' instead or another monitor"
            )
    else:
        raise ValueError(f"Unknown checkpoint: {checkpoint_path}")


def load_training(
    lit_model_class: pl.LightningModule,
    args: Namespace = None,
    run_id: str = None,
    project_run_id: str = None,
    no_run_id: bool = None,
    **kwargs,
):
    """Load training"""
    run_id = args.run_id if run_id is None and "run_id" in args else run_id
    project_run_id = args.project_run_id if project_run_id is None and "project_run_id" in args else project_run_id
    weights_path = getattr(args, "weights", None) if args is not None else None
    no_run_id = args.no_run_id if no_run_id is None and "no_run_id" in args else no_run_id

    if "weights" in kwargs and kwargs["weights"] is not None:  # Highest priority
        weights_path = kwargs["weights"]

    strict = True if "nostrict" not in args else not args.nostrict
    if run_id is not None and project_run_id is not None:
        run_id_project_dir = os.path.join(vb_folder(), f"project_{project_run_id}", run_id)
        ckpt_path = kwargs.get("checkpoint") or getattr(args, "checkpoint", "last.ckpt")  # Highest priority on kwargs
        monitor = kwargs.get("monitor") or getattr(args, "monitor", "val_loss")
        ckpt_path = checkpoint_handler(ckpt_path, run_id_project_dir, monitor)
        ckpt_path = os.path.join(run_id_project_dir, ckpt_path)
        if not os.path.exists(ckpt_path):
            raise Exception(f"Impossible to load the ckpt at the following destination:{ckpt_path}")
        print(f"Loading ckpt from {run_id} at {ckpt_path}")
        lit_model = lit_model_class.load_from_checkpoint(ckpt_path, strict=strict, args=args, **kwargs)
    elif weights_path is not None:
        if os.path.splitext(weights_path.lower())[1] == ".pth":
            lit_model = lit_model_class(args=args, **kwargs)
        elif os.path.splitext(weights_path.lower())[1] == ".ckpt":
            lit_model = lit_model_class.load_from_checkpoint(weights_path, strict=strict, args=args, **kwargs)
        else:
            raise Exception(f"Impossible to load the weights at the following destination:{weights_path}")
    elif no_run_id:
        lit_model = lit_model_class(args=args, **kwargs)
    else:
        raise Exception("--run_id (optionally --project_run_id) must be given to load the experiment.")

    return lit_model


def set_save_dir_config(key):
    """
    Create /home/USER/.aloception/alonet_config.json with path to log/weights save dir.
    """
    streaming_config = os.path.join(vb_folder(create_if_not_found=True), "alonet_config.json")
    if not os.path.exists(streaming_config):
        with open(streaming_config, "w") as f:  # Json init as empty config
            json.dump(dict(), f, indent=4)
    with open(streaming_config) as f:
        content = json.loads(f.read())
    key = f"{key}_save_dir"
    default_dir = vb_folder()
    default_dir_message = (
        f"Do you want to use the default dir {default_dir} ? (Y)es or Please write a new directory for {key}: "
    )
    if key not in content:
        save_dir = _user_prompt(f"{key} is not set in config file. " + default_dir_message)
        if save_dir.lower() in ["y", "yes"]:
            save_dir = default_dir
    content[key] = save_dir
    with open(streaming_config, "w") as f:  # Save new directory
        json.dump(content, f, indent=4)
    return content[key]


def get_dir_from_config(args, key):
    """
    Get the directory to save log/checkpoint from the config in
    /home/USER/.aloception/alonet_config.json or from args if the dir is set.
    This file will be created if not exist.

    Parameters
    ----------
    args: Namespace
    key: str
        log or checkpoint

    Return
    ------
    The directory to save log/checkpoint
    """
    assert key in ["log", "checkpoint"], "Value of key must be log or checkpoint"
    save_dir = args.log_save_dir if key == "log" else args.cp_save_dir

    # get dir from config.json file if save_dir is not Set. Create the file if not exist
    if save_dir is None:
        streaming_config = os.path.join(vb_folder(create_if_not_found=True), "alonet_config.json")
        if not os.path.exists(streaming_config):
            save_dir = set_save_dir_config(key)
        with open(streaming_config) as f:
            content = json.loads(f.read())
        key_dir = f"{key}_save_dir"
        if key_dir not in content:
            return set_save_dir_config(key)
        else:
            return content[key_dir]
    else:
        return save_dir


def get_expe_infos(project, expe_name, args=None):
    """
    Get the directories for the project and the experimentation
    A date suffix is added to the expe_name
    """
    expe_name = expe_name or args.expe_name
    if args is not None and not args.no_suffix:
        expe_name = "{}_{:%B-%d-%Y-%Hh-%M}".format(expe_name, datetime.datetime.now())
    project_dir = os.path.join(get_dir_from_config(args, "checkpoint"), f"project_{project}")
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
