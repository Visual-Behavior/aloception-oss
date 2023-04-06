import matplotlib.pyplot as plt
import lightning as pl
import numpy as np
import wandb


class RAFTEPECallback(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()
        self.datamodule = datamodule
        self.names = datamodule.val_names
        self.epe_per_iter = [None] * len(self.names)
        self.epoch = 0

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict,
        batch: list,
        batch_idx: int,
        dataloader_idx: int,
    ):

        epe_per_iter = outputs["epe_per_iter"]
        # first call
        if self.epe_per_iter[dataloader_idx] is None:
            self.epe_per_iter[dataloader_idx] = [[epe.cpu().numpy()] for epe in epe_per_iter]
        else:
            for it, epe in enumerate(epe_per_iter):
                self.epe_per_iter[dataloader_idx][it].append(epe.cpu().numpy())

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.logger is None:
            return

        for name, epe_per_it in zip(self.names, self.epe_per_iter):
            epe_per_it = [np.mean(np.concatenate(epe)) for epe in epe_per_it]
            log_iteration_epe(trainer, epe_per_it, self.epoch, name)

        self.epoch += 1
        self.epe_per_iter = [None] * len(self.names)


def log_iteration_epe(trainer: pl.Trainer, errors, epoch, name):
    # log as an image
    title = f"EPE per Raft iteration [epoch:{epoch}]"
    x = np.arange(len(errors))
    y = np.array(errors)
    fig = plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("RAFT_iteration")
    plt.ylabel("EPE")
    image = wandb.Image(plt_fig_to_np_array(fig))
    trainer.logger.experiment.log(
        {f"{name}/errors_per_iter/epe_per_iter": image}, step=trainer.global_step, commit=False
    )
    plt.close()


def plt_fig_to_np_array(fig, remove_axis=False, tight=False):
    ax = fig.gca()
    if remove_axis:
        ax.axis("off")
    if tight:
        fig.tight_layout(pad=0)
        ax.margins(0)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image_from_plot
