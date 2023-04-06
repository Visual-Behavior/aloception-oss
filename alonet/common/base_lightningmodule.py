import lightning as pl
from argparse import Namespace, ArgumentParser
import torch
import alonet
from typing import Union
import aloscene


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

        self.model = self.build_model(weights=self.weights)
        self.criterion = self.build_criterion()

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

        return parent_parser

    def configure_optimizers(self):
        """AdamW default optimizer configuration, using different learning rates for backbone and others parameters
        Returns
        -------
        torch.optim
            `AdamW <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html>`_ optimizer to update weights
        """
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad]},
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)
        return optimizer

    def build_model(self, weights):
        raise NotImplementedError("Should be implemented in child class.")

    def build_criterion(self):
        raise NotImplementedError("Should be implemented in child class.")

    def forward(self, frames: Union[list, aloscene.Frame], **kwargs):
        """Run a forward pass through the model.
        Parameters
        ----------
        frames : Union[list, :mod:`Frames <aloscene.frame>`]
            List of :mod:`~aloscene.frame` without batch dimension or a Frame with the batch dimension
        Returns
        -------
        m_outputs: dict
            A dict with the forward outputs.
        """
        # Batch list of frame if needed
        if isinstance(frames, list):
            frames = aloscene.Frame.batch_list(frames)

        # Assert inputs content
        self.assert_input(frames, inference=True)

        m_outputs = self.model(frames)

        return m_outputs

    def inference(self, m_outputs: dict, frames: aloscene.Frame, **kwargs):
        """Given the model forward outputs, run inference.
        Parameters
        ----------
        m_outputs: dict
            Dict with the model forward outptus
        """
        raise NotImplementedError("Should be implemented in child class.")

    def training_step(self, frames: Union[list, aloscene.Frame], batch_idx: int):
        """Train the model for one step
        Parameters
        ----------
        frames : Union[list, :mod:`Frames <aloscene.frame>`]
            List of :mod:`~aloscene.frame` without batch dimension or a Frame with the batch dimension
        batch_idx : int
            Batch id given by Lightning
        Returns
        -------
        dict
            Dictionary with the :attr:`loss` to optimize, :attr:`m_outputs` forward outputs and :attr:`metrics` to log.
        """
        if isinstance(frames, list):
            frames = aloscene.Frame.batch_list(frames)

        # Assert inputs content
        self.assert_input(frames)

        m_outputs = self.model(frames)

        loss, metrics = self.criterion(frames, m_outputs)

        outputs = {"loss": loss}
        outputs.update({"m_outputs": m_outputs})
        outputs.update({"metrics": metrics})

        return outputs

    @torch.no_grad()
    def validation_step(self, frames: Union[list, aloscene.Frame], batch_idx: int):
        """Run one step of validation
        Parameters
        ----------
        frames : Union[list, :mod:`Frames <aloscene.frame>`]
            List of :mod:`~aloscene.frame` without batch dimension or a Frame with the batch dimension
        batch_idx : int
            Batch id given by Lightning
        Returns
        -------
        dict
            Dictionary with the :attr:`loss` to optimize, and :attr:`metrics` to log.
        """
        # Batch list of frame if needed
        with torch.no_grad():
            if isinstance(frames, list):
                frames = aloscene.Frame.batch_list(frames)

            # Assert inputs content
            self.assert_input(frames)

            m_outputs = self.model(frames)

            loss, metrics = self.criterion(frames, m_outputs)

            outputs = {"loss": loss}
            outputs.update({"metrics": metrics})

            return outputs

    def assert_input(self, frames: aloscene.Frame, inference=False):
        """Check if input-frames have the correct format
        Parameters
        ----------
        frames : :mod:`Frames <aloscene.frame>`
            Input frames
        inference : bool, optional
            Check input from inference procedure, by default False
        """
        raise NotImplementedError("Should be implemented in child class.")

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

    def run_validation(
        self,
        data_loader: torch.utils.data.DataLoader,
        args: Namespace = None,
        project: str = None,
        expe_name: str = None,
        callbacks: list = None,
    ):
        """
        Validate the model using pytorch lightning
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

        alonet.common.pl_helpers.run_pl_validate(
            # Trainer, data & callbacks
            lit_model=self,
            data_loader=data_loader,
            callbacks=callbacks,
            # Project info
            args=args,
            project=project,
        )
