import aloscene
import numpy as np


def main():

    labels = aloscene.Labels([0, 1, 1])
    points = aloscene.Points2D([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], points_format="xy", absolute=False, labels=labels)
    points.get_view().render()

    # frame.get_view().render()


if __name__ == "__main__":
    main()
