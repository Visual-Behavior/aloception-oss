import pytorch_lightning as pl
import torch
from typing import Union, List

import alonet
import aloscene


class BaseLightningModule(pl.LightningModule):
    """BaseLightningModule"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = self.build_model()
        self.criterion = self.build_criterion()

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, dict]:
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

    def build_model(self):
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

    def callbacks(self, **kwargs) -> List[pl.Callback]:
        """This method will return the default callbacks of the training loop.
        Returns
        -------
        List[:doc:`alonet.callbacks`]
            Callbacks use in train process
        """

        metrics_callback = alonet.callbacks.MetricsCallback()
        return [metrics_callback]
