import pytorch_lightning as pl
import torch.optim as optim

from alonet.callbacks import MetricsCallback
from alonet.raft.criterion import RAFTCriterion
from alonet.raft.callbacks import RAFTFlowImagesCallback, RAFTEPECallback, FlowVideoCallback
from aloscene import Frame
import alonet


class LitRAFT(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.weights = args.weights
        self.model = self.build_model(weights=args.weights)
        self.criterion = self.build_criterion()

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("LitRAFT")
        parser.add_argument("--weights", type=str, help="for example raft-things")
        # override pl.Trainer gradient_clip default value
        parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
        return parent_parser

    def forward(self, frames, only_last=True):
        # prepare data
        if isinstance(frames, list):
            frames = Frame.batch_list(frames)
        self.assert_input(frames, inference=True)
        frame1 = frames[:, 0, ...]
        frame2 = frames[:, 1, ...]
        # run forward pass model
        m_outputs = self.model(frame1, frame2, only_last=only_last)
        return m_outputs

    def inference(self, m_outputs, only_last=True):
        return self.model.inference(m_outputs, only_last=only_last)

    def training_step(self, frames, batch_idx):
        # prepare data
        if isinstance(frames, list):
            frames = Frame.batch_list(frames)
        self.assert_input(frames, inference=False)
        frame1 = frames[:, 0, ...]
        frame2 = frames[:, 1, ...]
        # run forward pass model
        m_outputs = self.model(frame1, frame2, only_last=False)
        flow_loss, metrics, epe_per_iter = self.criterion(m_outputs, frame1)
        outputs = {"loss": flow_loss, "metrics": metrics, "epe_per_iter": epe_per_iter}
        return outputs

    def validation_step(self, frames, batch_idx, dataloader_idx=None):
        # prepare data
        if isinstance(frames, list):
            frames = Frame.batch_list(frames)
        self.assert_input(frames, inference=True)
        frame1 = frames[:, 0, ...]
        frame2 = frames[:, 1, ...]
        # run forward pass model
        m_outputs = self.model(frame1, frame2, only_last=False)
        flow_loss, metrics, epe_per_iter = self.criterion(m_outputs, frame1, compute_per_iter=True)
        outputs = {"val_loss": flow_loss, "metrics": metrics, "epe_per_iter": epe_per_iter}
        return outputs

    def build_criterion(self):
        return RAFTCriterion()

    def build_model(self, weights=None, device="cpu", dropout=0):
        return alonet.raft.RAFT(weights=weights, device=device, dropout=dropout)

    def configure_optimizers(self, lr=4e-4, weight_decay=1e-4, epsilon=1e-8, numsteps=100000):
        params = self.model.parameters()
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay, eps=epsilon)
        if self.args.max_steps is None:
            return optimizer
        else:
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                lr,
                self.args.max_steps + 100,
                pct_start=0.05,
                cycle_momentum=False,
                anneal_strategy="linear",
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def assert_input(self, frames, inference=False):
        assert (
            frames.normalization == "minmax_sym"
        ), f"frames.normalization should minmax_sym, not '{frames.normalization}'"
        assert frames.names == (
            "B",
            "T",
            "C",
            "H",
            "W",
        ), f"frames.names should be ('B','T','C','H','W'), not: '{frames.names}'"

        if inference:
            return

        assert frames.flow is not None, "A flow label should be attached to the frame"

    def callbacks(self, data_loader):
        """Given a data_loader, this method will return the default callbacks
        of the training loop.
        """

        metrics_callback = MetricsCallback(val_names=data_loader.val_names)
        flow_images_callback = RAFTFlowImagesCallback(data_loader)
        # flow_video_callback = FlowVideoCallback(data_loader, max_frames=30, fps=3)
        flow_epe_callback = RAFTEPECallback(data_loader)
        return [metrics_callback, flow_images_callback, flow_epe_callback]

    def run_train(self, data_loader, args, project="raft", expe_name="raft", callbacks: list = None):
        """Train the model using pytorch lightning"""
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

    def run_validation(self, data_loader, args, project="raft", expe_name="raft", callbacks: list = None):
        """Validate the model using pytorch lightning"""
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
            expe_name=expe_name,
        )
