import pytorch_lightning as pl
from argparse import Namespace, ArgumentParser
import torch
import alonet


class BaseLightningModule(pl.LightningModule):
    """
    Parameters
    ----------
    args : Namespace, optional
        Attributes stored in specific Namespace, by default None
    weights : str, optional
        Weights name to load, by default None
    gradient_clip_val : float, optional
        pytorch_lightning.trainer.trainer parameter. 0 means don’t clip, by default 0.1
    accumulate_grad_batches : int, optional
        Accumulates grads every k batches or as set up in the dict, by default 4
    model_name : str, optional
        Name use to define the model, by default "detr-r50"
    model : torch.nn, optional
        Custom model to train
    Notes
    -----
    Arguments entered by the user (kwargs) will replace those stored in args attribute
    """

    def __init__(self, args: Namespace = None, model: torch.nn = None, **kwargs):
        super().__init__()
        # Update class attributes with args and kwargs inputs
        alonet.common.pl_helpers.params_update(self, args, kwargs)
        self._init_kwargs_config.update({"model": model})

        # # Load model
        # if model is not None:
        #     if isinstance(model, str):
        #         self.model_name = model
        #         self.model = self.build_model()
        #     elif self.weights is None:
        #         self.model = model
        #     else:
        #         raise Exception(f"Weights of custom model doesnt match with {self.weights} weights")
        # else:
        #     self.model = self.build_model()
        # # Buld matcher
        # self.matcher = self.build_matcher()
        # # Build criterion
        # self.criterion = self.build_criterion(matcher=self.matcher)

    @staticmethod
    def add_argparse_args(parent_parser, parser=None):
        """Add arguments to parent parser with default values
        Parameters
        ----------
        parent_parser : ArgumentParser
            Object to append new arguments
        parser : ArgumentParser.argument_group, optional
            Argument group to append the parameters, by default None
        Returns
        -------
        ArgumentParser
            Object with new arguments concatenated
        """
        parser = parent_parser.add_argument_group("BaseLightningModule") if parser is None else parser
        parser.add_argument("--weights", type=str, default=None, help="One of (detr-r50). (Default: %(default)s)")
        parser.add_argument(
            "--gradient_clip_val", type=float, default=0.1, help="Gradient clipping norm (Default: %(default)s)"
        )
        parser.add_argument(
            "--accumulate_grad_batches",
            type=int,
            default=4,
            help="Number of gradient accumulation steps (Default: %(default)s)",
        )
        # TODO : add project and expe_name if not already present ?
        return parent_parser

    def callbacks(self,):
        """Given a data loader, this method will return the default callbacks of the training loop.

        Returns
        -------
        List[:doc:`alonet.callbacks`]
            Callbacks use in train process
        """

        metrics_callback = alonet.callbacks.MetricsCallback()
        ap_metrics_callback = alonet.callbacks.ApMetricsCallback()
        return [metrics_callback, ap_metrics_callback]

    def run_train(
        self,
        data_loader: torch.utils.data.DataLoader,
        args: Namespace = None,
        project: str = None,
        expe_name: str = None,
        callbacks: list = None,
    ):
        """Train the model using pytorch lightning
        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Dataloader use in :func:`callbacks` function
        project : str, optional
            Project name using to save checkpoints, by default None
        expe_name : str, optional
            Specific experiment name to save checkpoints, by default None
        callbacks : list, optional
            List of callbacks to use, by default :func:`callbacks` output
        args : Namespace, optional
            Additional arguments use in training process, by default None
        """
        # Set the default callbacks if not provide.
        callbacks = callbacks if callbacks is not None else self.callbacks(data_loader)

        alonet.common.pl_helpers.run_pl_training(
            # Trainer, data & callbacks
            lit_model=self,
            data_loader=data_loader,
            callbacks=callbacks,
            # Project info
            args=args,
            project=project,
            expe_name=expe_name,
        )


# TODO : add run_validate
