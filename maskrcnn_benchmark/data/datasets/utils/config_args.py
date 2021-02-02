# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import os.path as op

def config_tsv_dataset_args(cfg, dataset_file, factory_name=None, is_train=True):
    full_yaml_file = op.join(cfg.DATA_DIR, dataset_file)

    assert op.isfile(full_yaml_file)

    extra_fields = ["class"]
    if cfg.MODEL.MASK_ON:
        extra_fields.append("mask")
    if cfg.MODEL.ATTRIBUTE_ON:
        extra_fields.append("attributes")

    skip_performance_eval = False if is_train else cfg.TEST.SKIP_PERFORMANCE_EVAL
    is_load_label = not skip_performance_eval

    args = dict(
        yaml_file=full_yaml_file,
        extra_fields=extra_fields,
        is_load_label=is_load_label,
    )

    if factory_name is not None:
        tsv_dataset_name = factory_name
    else:
        if "openimages" in dataset_file:
            tsv_dataset_name = "OpenImagesVRDTSVDataset"
        elif "visualgenome" in dataset_file or "gqa" in dataset_file:
            tsv_dataset_name = "VGTSVDataset"

    args['args'] = cfg

    return args, tsv_dataset_name
