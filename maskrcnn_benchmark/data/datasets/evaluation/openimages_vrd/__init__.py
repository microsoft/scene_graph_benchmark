# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
from .openimages_vrd_eval import do_openimages_vrd_evaluation


def openimages_vrd_evaluation(dataset, output_folder, force_relation=False, **kwargs):
    return do_openimages_vrd_evaluation(
        dataset=dataset,
        output_folder=output_folder,
        force_relation=force_relation,
    )
