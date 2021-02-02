# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import logging

# from .sg_eval import do_sg_evaluation
from .sg_tsv_eval import do_sg_evaluation


def sg_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("scene_graph_generation.inference")
    logger.warning("performing scene graph evaluation.")
    return do_sg_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )
