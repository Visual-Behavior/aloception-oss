Metrics
======================

AP Metrics
----------------------------------

Stores all the information necessary to calculate the AP for one IoU and one class.

Examples
^^^^^^^^
.. code-block:: python

    import torch

    from alodataset import CocoBaseDataset
    from alonet.detr import DetrR50
    from alonet.metrics import ApMetrics

    from aloscene import Frame

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Model/Dataset definition to evaluate
    dataset = CocoBaseDataset(sample=True)
    model = DetrR50(weights = "detr-r50")
    model.eval().to(device)

    # Metric to develop
    metric = ApMetrics()

    for frame in dataset.stream_loader():
        # Frame to batch
        frame = Frame.batch_list([frame]).to(device)

        # Boxes inference and get GT from frame
        pred_boxes = model.inference(model(frame))[0]  # Predictions of the first batch
        gt_boxes = frame[0].boxes2d  # GT of first batch

        # Add samples to evaluate metrics
        metric.add_sample(p_bbox=pred_boxes, t_bbox=gt_boxes)

    # Print results
    metric.calc_map(print_result=True)

.. list-table:: Results obtained for AP in boxes
    :header-rows: 1
    :align: center

    * -
      - all
      - .50
      - .55
      - .60
      - .65
      - .70
      - .75
      - .80
      - .85
      - .90
      - .95
    * - box
      - 40.21
      - 49.98
      - 49.12
      - 47.68
      - 46.66
      - 44.23
      - 43.89
      - 36.38
      - 33.78
      - 30.14
      - 20.20

.. automodule:: alonet.metrics.compute_map
   :members:
   :undoc-members:
   :show-inheritance:

PQ Metrics
----------------------------------

Compute Panoptic, Segmentation and Recognition Qualities Metrics.

Examples
^^^^^^^^
.. code-block:: python

    import torch

    from alodataset import CocoPanopticDataset, Split
    from alonet.detr_panoptic import LitPanopticDetr
    from alonet.metrics import PQMetrics

    from aloscene import Frame

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Model/Dataset definition to evaluate
    dataset = CocoPanopticDataset(split=Split.VAL)
    model = LitPanopticDetr(weights = "detr-r50-panoptic").model
    model.eval().to(device)

    # Metric to develop
    metric = PQMetrics()

    for frame in dataset.stream_loader():
        # Frame to batch
        frame = Frame.batch_list([frame]).to(device)

        # Masks inference and get GT from frame
        _, pred_masks = model.inference(model(frame))
        pred_masks = pred_masks[0] # Predictions of the first batch
        gt_masks = frame[0].segmentation  # GT of first batch

        # Add samples to evaluate metrics
        metric.add_sample(p_mask=pred_masks, t_mask=gt_masks)

    # Print results
    metric.calc_map(print_result=True)

.. list-table:: Summary of expected result
    :widths: 15 15 10 10 10
    :header-rows: 1
    :align: center

    * - Category
      - Total cat.
      - PQ
      - SQ
      - RQ
    * - Things
      - 80
      - 0.481
      - 0.795
      - 0.595
    * - Stuff
      - 53
      - 0.358
      - 0.779
      - 0.449
    * - All
      - 133
      - 0.432
      - 0.789
      - 0.537

.. automodule:: alonet.metrics.compute_pq
   :members:
   :undoc-members:
   :show-inheritance:
