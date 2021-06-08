# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
from maskrcnn_benchmark.data import datasets

from .coco import coco_evaluation
from .voc import voc_evaluation
from .cityscapes import abs_cityscapes_evaluation
from .sg import sg_evaluation
from .openimages_vrd import openimages_vrd_evaluation
from .vg import vg_evaluation


def evaluate(dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )
    if isinstance(dataset, datasets.COCODataset):
        return coco_evaluation(**args)
    elif isinstance(dataset, datasets.PascalVOCDataset):
        return voc_evaluation(**args)
    elif isinstance(dataset, datasets.OpenImagesVRDTSVDataset):
        return openimages_vrd_evaluation(**args)
    elif isinstance(dataset, datasets.VGTSVDataset):
        if 'sg_eval' in args and args['sg_eval']:
            return sg_evaluation(**args)
        else:
            return vg_evaluation(**args)
    elif isinstance(dataset, datasets.AbstractDataset):
        return abs_cityscapes_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
