# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import json

import torch
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.data.datasets.utils.load_files import config_dataset_file
from maskrcnn_benchmark.engine.inference import inference
from scene_graph_benchmark.scene_parser import SceneParser
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def run_test(cfg, model, distributed, model_name):
    if distributed and hasattr(model, 'module'):
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        if len(dataset_names) == 1:
            output_folder = os.path.join(
                cfg.OUTPUT_DIR, "inference",
                os.path.splitext(model_name)[0]
            )
            mkdir(output_folder)
            output_folders = [output_folder]
        else:
            for idx, dataset_name in enumerate(dataset_names):
                dataset_name1 = dataset_name.replace('/', '_')
                output_folder = os.path.join(
                    cfg.OUTPUT_DIR, "inference",
                    dataset_name1,
                    os.path.splitext(model_name)[0]
                )
                mkdir(output_folder)
                output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    labelmap_file = config_dataset_file(cfg.DATA_DIR, cfg.DATASETS.LABELMAP_FILE)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        results = inference(
            model,
            cfg,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            bbox_aug=cfg.TEST.BBOX_AUG.ENABLED,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            skip_performance_eval=cfg.TEST.SKIP_PERFORMANCE_EVAL,
            labelmap_file=labelmap_file,
            save_predictions=cfg.TEST.SAVE_PREDICTIONS,
        )

        # renaming box_proposals metric to rpn_proposals if RPN_ONLY is True
        if results and 'box_proposal' in results and cfg.MODEL.RPN_ONLY:
            results['rpn_proposal'] = results.pop('box_proposal')

        if results and output_folder:
            results_path = os.path.join(output_folder, "results.json")
            # checking if this file already exists and only updating tasks
            # that are already present. This is useful for including
            # e.g. RPN_ONLY metrics
            if os.path.isfile(results_path):
                with open(results_path, 'rt') as fin:
                    old_results = json.load(fin)
                old_results.update(results)
                results = old_results
            with open(results_path, 'wt') as fout:
                json.dump(results, fout)

        synchronize()

    # evaluate attribute detection
    if not cfg.MODEL.RPN_ONLY and cfg.MODEL.ATTRIBUTE_ON and (not cfg.TEST.SKIP_PERFORMANCE_EVAL):
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
        for output_folder, dataset_name, data_loader_val in zip(
            output_folders, dataset_names, data_loaders_val
        ):
            results_attr = inference(
                model,
                cfg,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
                skip_performance_eval=cfg.TEST.SKIP_PERFORMANCE_EVAL,
                labelmap_file=labelmap_file,
                save_predictions=cfg.TEST.SAVE_PREDICTIONS,
                eval_attributes=True,
            )

            if results_attr and output_folder:
                results_path = os.path.join(output_folder, "results.json")
                # checking if this file already exists and only updating tasks
                # that are already present. This is useful for including
                # e.g. RPN_ONLY metrics
                if os.path.isfile(results_path):
                    with open(results_path, 'rt') as fin:
                        old_results = json.load(fin)
                    old_results.update(results_attr)
                    results_attr = old_results
                with open(results_path, 'wt') as fout:
                    json.dump(results_attr, fout)

            synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend=cfg.DISTRIBUTED_BACKEND, init_method="env://"
        )
        synchronize()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    if cfg.MODEL.META_ARCHITECTURE == "SceneParser":
        model = SceneParser(cfg)
    elif cfg.MODEL.META_ARCHITECTURE == "AttrRCNN":
        model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)
    model_name = os.path.basename(ckpt)

    run_test(cfg, model, args.distributed, model_name)


if __name__ == "__main__":
    main()
