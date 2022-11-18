import aloscene
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import warnings
import numpy as np
import wandb
from alodataset.utils.panoptic_utils import id2rgb


def boxes_to_wandb_boxes(boxes: aloscene.BoundingBoxes2D, labels_names: list = None):
    """Convert a set of boxes to wandb boxex

    Parameters
    ----------
    boxes: aloscene.BoundingBoxes2D
        Set of boxes to convert to wandb boxes format
    labels_names: list
        Dictionnary mapping each label id to the label name
    """
    boxes = boxes.xyxy().rel_pos()
    boxes_labels = None if boxes.labels is None else boxes.labels.detach().cpu()
    scores = boxes_labels.scores if boxes_labels is not None else None

    box_data = []

    for b, box in enumerate(boxes):
        box = box.detach().cpu().as_tensor()
        n_box = {
            # one box expressed in the default relative/fractional domain
            "position": {"minX": float(box[0]), "maxX": float(box[2]), "minY": float(box[1]), "maxY": float(box[3])}
        }
        if boxes_labels is not None:
            n_box["class_id"] = int(boxes_labels[b])
            if labels_names is not None and (n_box["class_id"] < len(labels_names) and n_box["class_id"] >= 0):
                n_box["box_caption"] = labels_names[int(boxes_labels[b])]
            else:
                n_box["box_caption"] = str(int(boxes_labels[b]))
        else:
            n_box["class_id"] = -1
            n_box["box_caption"] = "unknown"

        if scores is not None:
            n_box["scores"] = {"score": float(scores[b])}
        else:
            n_box["scores"] = {"score": float(1)}

        box_data.append(n_box)

    return box_data


def log_scalar(trainer, key, obj):
    """Log a scalar to the current logger

    Parameters
    ----------
    trainer: pytorch_lightning.Trainer
        The pytorch_lightning current trainer
    key: str
        Tag name of the log
    obj: torch.Tensor
        scalar tensor
    """
    if isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.log({key: obj, "trainer/global_step": trainer.global_step})
    elif isinstance(trainer.logger, TensorBoardLogger):
        trainer.logger.experiment.add_scalar(key, obj, trainer.global_step)
    else:
        warnings.warn("image logging not implemented for current logger")


def log_image(trainer, key, images):
    """Log an image to the current logger

    Parameters
    ----------
    trainer: pytorch_lightning.Trainer
        The pytorch_lightning current trainer
    key: str
        Tag name of the log
    images: list of dict]
        List of dict with
        `image` : torch.Tensor
        `boxes`: aloscene.BoundingBoxes2D, optional
        `masks`: aloscene.Mask, optional
    """
    if isinstance(trainer.logger, WandbLogger):
        wandb_images = []
        for i, image_data in enumerate(images):
            image = image_data["image"]
            boxes = None if "boxes" not in image_data else image_data["boxes"]
            masks = None if "masks" not in image_data else image_data["masks"]
            boxes_dict, masks_dict = None, None
            if boxes is not None:
                boxes_dict = {}
                for i, b in enumerate(boxes):
                    boxes_dict[b["name"]] = {"box_data": boxes_to_wandb_boxes(b["boxes"], b["class_labels"])}
                    if b["class_labels"] is not None:
                        boxes_dict[b["name"]]["class_labels"] = b["class_labels"]
            if masks is not None:
                masks_dict = {}
                for i, m in enumerate(masks):
                    masks_dict[m["name"]] = {"mask_data": m["masks"]}
                    if m["class_labels"] is not None:
                        masks_dict[m["name"]]["class_labels"] = m["class_labels"]
            wandb_images.append(wandb.Image(image, boxes=boxes_dict, masks=masks_dict))

        trainer.logger.experiment.log({key: wandb_images, "trainer/global_step": trainer.global_step})

    elif isinstance(trainer.logger, TensorBoardLogger):

        for i, image_obj in enumerate(images):
            batch_el_key = f"{key}_{i}"
            image = image_obj["image"]
            boxes = None if "boxes" not in image_obj else image_obj["boxes"]
            masks = None if "masks" not in image_obj else image_obj["masks"]

            if boxes is None and masks is None:
                image = image.transpose(2, 0, 1)
                trainer.logger.experiment.add_image(batch_el_key, image, trainer.global_step)

            if masks is not None:
                for m in masks:
                    img = (256 * id2rgb(m["masks"]) * 0.8 + image * 0.2).transpose(2, 0, 1).astype(np.uint8)
                    trainer.logger.experiment.add_image(f"{batch_el_key}_{m['name']}", img, trainer.global_step)
            if boxes is not None:
                image = aloscene.Frame(np.transpose(image, (2, 0, 1)), names=["C", "H", "W"])
                for b in boxes:
                    img = b["boxes"].get_view(image).image
                    img = np.transpose(img, (2, 0, 1))
                    trainer.logger.experiment.add_image(f"{batch_el_key}_{b['name']}", img, trainer.global_step)
    else:
        warnings.warn("image logging not implemented for current logger")


def log_figure(trainer, key, obj):
    """Log a matplotlib figure to the current logger

    Parameters
    ----------
    trainer: pytorch_lightning.Trainer
        The pytorch_lightning current trainer
    key: str
        Tag name of the log
    obj: matplotlib.pyplot.Figure
        The matplotlib Figure objectif to log
    """
    if isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.log({key: wandb.Image(obj), "trainer/global_step": trainer.global_step})
    elif isinstance(trainer.logger, TensorBoardLogger):
        trainer.logger.experiment.add_figure(key, obj, trainer.global_step)
    else:
        warnings.warn("figure logging not implemented for current logger")


def log_scatter(trainer, key, values, column_names):
    """Log a scatter graph to the current logger

    Parameters
    ----------
    trainer: pytorch_lightning.Trainer
        The pytorch_lightning current trainer
    key: str
        Tag name of the log
    values: torch.Tensor
        values of the graphe tensor (N, 2)
    column_names:List[str]
        names of axes (x axis, y axis)
    """
    if isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.log(
            {
                key: wandb.plot.scatter(
                    wandb.Table(data=values, columns=column_names), x=column_names[0], y=column_names[1], title=key
                ),
                "trainer/global_step": trainer.global_step,
            }
        )
    else:
        warnings.warn("scatter logging not implemented for current logger")


def log_hist(trainer, key, hist):
    """Log a histogram graph to the current logger

    Parameters
    ----------
    trainer: pytorch_lightning.Trainer
        The pytorch_lightning current trainer
    key: str
        Tag name of the log
    values: torch.Tensor
        values of the graphe tensor (N, 2)
    hist: torch.tensor
        names of axes (x axis, y axis)
    """
    hist = hist.cpu()
    if isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.log({key: wandb.Histogram(hist), "trainer/global_step": trainer.global_step})
    elif isinstance(trainer.logger, TensorBoardLogger):
        trainer.logger.experiment.add_histogram(key, hist, trainer.global_step)
    else:
        warnings.warn("hist logging not implemented for current logger")


def log_hyperparams(trainer, dict):
    """Log hyperparameters to the current logger

    Parameters
    ----------
    trainer: pytorch_lightning.Trainer
        The pytorch_lightning current trainer
    dict : Namespace or Dict containing the hyperparameters
    """
    if isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.log_hyperparams(dict)

    elif isinstance(trainer.logger, TensorBoardLogger):
        warnings.warn("hyperparameters logging not implemented for current logger")
