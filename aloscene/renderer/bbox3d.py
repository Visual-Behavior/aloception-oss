# from __future__ import annotations

from typing import *
import numpy as np
import cv2


def _draw_3D_box(image, vertices_2d, color1=None, color2=None, class_names=None, labels=None, colors=None):
    """
    Parametres
    ----------
    vertices_2d of shape (nbox, 8, 2)
    """
    for i in range(vertices_2d.shape[0]):
        label = labels[i] if labels is not None and len(labels) > 0 else None
        if class_names is None:
            label_name = "unknow"
        else:
            label_name = class_names[int(labels[i])]
        # class_color = CLASS_COLOR_MAP[int(labels[i])]

        points = np.clip(vertices_2d[i].astype(np.int), 0, np.max(image.shape))

        if color1 is None:
            # color1 = (0, 255, 0)
            color1 = (0, 1.0, 0)
        if color2 is None:
            # color2 = (0, 0, 255)
            color2 = (0, 0, 1.0)
        # overwrite color is colors is defined
        if colors is not None:
            color1 = colors[i]
            color2 = colors[i] ** 2
        # plt.imshow(image)
        # plt.show()
        for i in range(4):
            point_1_ = points[2 * i]
            point_2_ = points[2 * i + 1]
            cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), color1, 2)
            # plt.imshow(image)
            # plt.show()

        for i in range(4):
            point_1_ = points[i]
            point_2_ = points[i + 4]
            cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), color1, 2)
            # plt.imshow(image)
            # plt.show()

        for i in [0, 1, 4, 5]:
            point_1_ = points[i]
            point_2_ = points[i + 2]
            cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), color1, 2)
            # plt.imshow(image)
            # plt.show()

        point_1_ = points[0]
        point_2_ = points[3]
        cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), color2, 2)
        # plt.imshow(image)
        # plt.show()

        point_1_ = points[2]
        point_2_ = points[1]
        cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), color2, 2)
        # plt.imshow(image)
        # plt.show()

        if class_names is not None and label is not None:
            point_1_ = points[4]
            point_2_ = points[6]
            cv2.putText(
                image,
                str(label_name),
                (point_1_[0] + ((point_2_[0] - point_1_[0]) // 2), point_1_[1] + ((point_2_[1] - point_1_[1]) // 2)),
                # (point_1_[0], point_1_[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (1.0, 0, 0),
                1,
                cv2.LINE_AA,
            )


def draw_shiny_3D_box(image, vertices_2d, class_names=None, labels=None, colors=None):
    """
    Parameters
    ----------
    vertices_2d of shape (nbox, 8, 2)
    """

    def draw_gradient_face(points, color):
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

        if (min_x > image.shape[1] - 1) or (min_y > image.shape[0] - 1):
            return
        if max_x < 0 or max_y < 0:
            return

        gradient = np.linspace(0, 1, num=max_y - min_y, endpoint=True, retstep=False, dtype=None, axis=0)  # (h, )
        gradient = gradient**4

        gradient = np.expand_dims(gradient, axis=-1)
        gradient = np.tile(gradient, [1, max_x - min_x])  # (h, w)

        gradient_selector = np.zeros((gradient.shape[0], gradient.shape[1], 3), dtype=np.float32)  # (h, w, 3)

        shift_points = points - np.array([min_x, min_y])
        cv2.fillPoly(gradient_selector, [shift_points], (255, 255, 255))
        selector = gradient_selector[:, :, 0] == 255  # (h*w)
        gradient_selector[selector] = np.expand_dims(gradient[selector], axis=-1) * [1, 1, 1]

        x1 = max(0, min_x)
        y1 = max(0, min_y)
        x2 = min(image.shape[1] - 1, max_x)
        y2 = min(image.shape[0] - 1, max_y)

        gy1 = y1 - min_y
        gx1 = x1 - min_x
        gy2 = y2 - y1 + gy1
        gx2 = x2 - x1 + gx1

        image[y1:y2, x1:x2] = (1 - gradient_selector[gy1:gy2, gx1:gx2]) * image[y1:y2, x1:x2] + (
            gradient_selector[gy1:gy2, gx1:gx2] * color
        )

    for i in range(vertices_2d.shape[0]):

        if class_names is None or len(class_names) == 0:
            label_name = "unknown"
        else:
            label_name = class_names[int(labels[i])]

        class_color = colors[i] if (colors is not None and len(colors) > 0) else [1, 1, 1]

        label = labels[i] if labels is not None and len(labels) > 0 else None
        points = vertices_2d[i].astype(np.int)
        n_points = np.concatenate([[points[0]], [points[4]], [points[6]], [points[2]]], axis=0)
        draw_gradient_face(n_points, color=class_color)
        n_points = np.concatenate([[points[0]], [points[1]], [points[5]], [points[4]]], axis=0)
        draw_gradient_face(n_points, color=class_color)
        n_points = np.concatenate([[points[2]], [points[6]], [points[7]], [points[3]]], axis=0)
        draw_gradient_face(n_points, color=class_color)
        n_points = np.concatenate([[points[3]], [points[1]], [points[5]], [points[7]]], axis=0)
        draw_gradient_face(n_points, color=class_color)

        if class_names is not None and label is not None:
            point_1_ = points[4]
            point_2_ = points[6]
            cv2.putText(
                image,
                str(label_name),
                (point_1_[0] + ((point_2_[0] - point_1_[0]) // 2), point_1_[1] + ((point_2_[1] - point_1_[1]) // 2)),
                # (point_1_[0], point_1_[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )


def draw_3D_box(
    image, vertices_2d, color1=None, color2=None, class_names=None, labels=None, colors=None, draw_shiny=False
):
    _draw_3D_box(image, vertices_2d, color1, color2, class_names, labels, colors)
    if draw_shiny:
        draw_shiny_3D_box(image, vertices_2d, class_names, labels, colors)
