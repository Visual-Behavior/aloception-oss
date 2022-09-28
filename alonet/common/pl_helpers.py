from argparse import ArgumentParser, ArgumentTypeError, Namespace, _ArgumentGroup
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import datetime
import os
import re
import torch

parser = ArgumentParser()


def vb_folder():
    home = os.getenv("HOME")
    alofolder = os.path.join(home, ".aloception")
    if not os.path.exists(alofolder):
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


def add_argparse_args(parent_parser, add_pl_args=True, mode="training"):
    """add cli argparse arguments to parent_parser

    Parameters
    ----------
    parent_parser: argparse.ArgumentParser
            The custom cli arguments parser, which will be extended by
            the class's default arguments.
    add_pl_args: bool
            By default, this is True, and also add pl.trainer arguments
            groups to the current parser
    mode : str, {"training", "eval"}
        Add args for training or eval

    Returns
    -------
    parent_parser: argparse.ArgumentParser
             original parent_parser with added arguments

    Raises
    ------
    RuntimeError:
        If ``parent_parser`` is not an ``ArgumentParser`` instance
    """
    if mode not in ["training", "eval"]:
        raise ValueError(f"Unknown value for parameter `mode`: {mode}")
    if isinstance(parent_parser, _ArgumentGroup):
        raise RuntimeError("Please only pass an ArgumentParser instance.")

    parser = parent_parser.add_argument_group("pl_helper")

    if add_pl_args:
        parent_parser = pl.Trainer.add_argparse_args(parent_parser)

    if mode == "training":
        parser.add_argument("--resume", action="store_true", help="Resume all the training, not only the weights.")
        parser.add_argument("--save", action="store_true", help="Save every epoch and keep the top_k")
        parser.add_argument(
            "--save_top_k",
            type=int,
            default=3,
            help="Stores up to top_k models, by default %(default)s. Use save_top_k=-1 to store all checkpoints",
        )
    if mode == "eval":  # Only make sense for eval
        parser.add_argument(
            "--checkpoint",
            type=checkpoint_type,
            default="last",
            help="Load the weights from 'best', 'last' or '.ckpt' file into run_id, by default '%(default)s'",
        )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        nargs="?",
        const="wandb",
        help="Log results, can specify logger, by default %(default)s. If set but value not provided:wandb",
    )
    parser.add_argument("--cpu", action="store_true", help="Use the CPU instead of scaling on the vaiable GPUs")
    parser.add_argument("--run_id", type=str, help="Load the weights from this saved experiment")
    parser.add_argument(
        "--monitor", type=str, default="val_loss", help="Metric to save/load weights, by default '%(default)s'"
    )
    parser.add_argument(
        "--no_run_id", action="store_true", help="Skip loading form run_id when an experiment is restored."
    )
    parser.add_argument("--project_run_id", type=str, help="Project related with the run ID to load")
    parser.add_argument("--expe_name", type=str, default=None, help="expe_name to be logged in wandb")
    parser.add_argument("--no_suffix", action="store_true", help="do not add date suffix to expe_name")
    parser.add_argument(
        "--nostrict",
        action="store_true",
        help="load from checkpoint to run a model with different weights names (default False)",
    )

    return parent_parser


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
                ck_props = dict(map(lambda y: y.split("="), re.findall("[-]*(.*?)(?:-|.ckpt)", fpath)))
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
    lit_model_class,
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


def get_expe_infos(project, expe_name, args=None):
    """
    Get the directories for the project and the experimentation
    A date suffix is added to the expe_name
    """
    expe_name = expe_name or args.expe_name
    if args is not None and not args.no_suffix:
        expe_name = "{}_{:%B-%d-%Y-%Hh-%M}".format(expe_name, datetime.datetime.now())
    project_dir = os.path.join(vb_folder(), f"project_{project}")
    expe_dir = os.path.join(project_dir, expe_name)
    return project_dir, expe_dir, expe_name


def run_pl_training(
    lit_model, data_loader, callbacks: list, args=None, project: str = None, expe_name: str = None, **pl_trainer
):
    """Helper to run training with pytorch lightning while handling project ID and names"""

    if args is None:
        args = Namespace()
        args.run_id = None
        args.project_run_id = None
        args.no_suffix = False
        args.log = None
        args.logger = "wandb"
        args.save = False
        args.cpu = False

    # Set the experiment name ID
    project_dir, expe_dir, expe_name = get_expe_infos(project, expe_name, args)
    resume_from_checkpoint = None

    if args.run_id is not None:
        strict = not args.nostrict
        run_id_project_dir = (
            project_dir if args.project_run_id is None else os.path.join(vb_folder(), f"project_{args.project_run_id}")
        )
        ckpt_path = os.path.join(run_id_project_dir, args.run_id, "last.ckpt")
        if not os.path.exists(ckpt_path):
            raise Exception(f"Impossible to load the ckpt at the following destination:{ckpt_path}")
        if not args.resume:
            kwargs_config = getattr(lit_model, "_init_kwargs_config", {})
            print(f"Loading ckpt from {args.run_id} at {ckpt_path}")
            lit_model = type(lit_model).load_from_checkpoint(ckpt_path, strict=strict, args=args, **kwargs_config)
        else:
            expe_name = args.run_id
            resume_from_checkpoint = ckpt_path
            expe_dir = os.path.join(run_id_project_dir, args.run_id)

    if args.log is not None:
        if args.log == "wandb":
            logger = WandbLogger(name=expe_name, project=project, id=expe_name)
        elif args.log == "tensorboard":
            logger = TensorBoardLogger(save_dir="tensorboard/", name=expe_name, sub_dir=expe_name)
        else:
            raise ValueError("Unknown or not implemented logger")
    else:
        logger = None

    if args.save:
        monitor = getattr(args, "monitor", "val_loss")
        checkpoint_callback = ModelCheckpoint(
            dirpath=expe_dir,
            verbose=True,
            save_last=True,
            save_top_k=getattr(args, "save_top_k", 3),
            monitor=monitor,
            filename="{epoch}-{step}-{" + monitor + ":.4f}",
        )
        callbacks.append(checkpoint_callback)

    # Init trainer and run training
    trainer = pl.Trainer.from_argparse_args(
        args,
        # default_root_dir=expe_dir,
        gpus=-1 if not args.cpu else 0,
        auto_select_gpus=not args.cpu,
        logger=logger,
        callbacks=callbacks,
        resume_from_checkpoint=resume_from_checkpoint,
        accelerator=None if torch.cuda.device_count() <= 1 else "ddp",
        **pl_trainer,
    )

    # Runing training
    trainer.fit(lit_model, data_loader)


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


def run_pl_validate(
    lit_model, data_loader, callbacks: list, args, project: str = None, expe_name: str = None, **pl_trainer
):
    """
    Helper to run a validation epoch with pytorch lightning while handling project ID and names
    """
    # Set the experiment name ID
    expe_name = get_expe_infos(project, expe_name, args)[-1]
    lit_model = load_training(type(lit_model), args, no_exception=True)

    # Init trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        # default_root_dir=expe_dir,
        gpus=-1 if not args.cpu else 0,
        auto_select_gpus=not args.cpu,
        logger=WandbLogger(name=expe_name, project=project, config=vars(args)) if args.log else None,
        callbacks=callbacks,
        **pl_trainer,
    )

    # validate
    trainer.validate(lit_model, data_loader.val_dataloader())


def _int_or_float_type(x):
    if "." in str(x):
        return float(x)
    return int(x)
