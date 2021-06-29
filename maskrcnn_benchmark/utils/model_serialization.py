# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict
import logging
import math

import torch

from maskrcnn_benchmark.utils.imports import import_file


def resize_pos_embed_1d(posemb, shape_new):
    # Rescale the grid of position embeddings when loading from state_dict.
    ntok_old = posemb.shape[1]
    if ntok_old > 1:
        ntok_new = shape_new[1]
        posemb_grid = posemb.permute(0, 2, 1).unsqueeze(dim=-1)
        posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=[ntok_new, 1], mode='bilinear')
        posemb_grid = posemb_grid.squeeze(dim=-1).permute(0, 2, 1)
        posemb = posemb_grid
    return posemb


def resize_pos_embed_2d(posemb, shape_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = shape_new[0]
    gs_old = int(math.sqrt(len(posemb)))  # 2 * w - 1
    gs_new = int(math.sqrt(ntok_new))  # 2 * w - 1
    posemb_grid = posemb.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(gs_new * gs_new, -1)
    return posemb_grid


def align_and_update_state_dicts(model_state_dict, loaded_state_dict, skip_unmatched_layers=True):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger(__name__)
    # print out no match
    uninitialized_keys = [current_keys[idx_new] for idx_new, idx_old in enumerate(idxs.tolist()) if idx_old == -1]
    logger.info("Parameters not initialized from checkpoint: {}\n".format(
        ','.join(uninitialized_keys)
    ))
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        if model_state_dict[key].shape != loaded_state_dict[
            key_old].shape and skip_unmatched_layers:
            if 'x_pos_embed' in key or 'y_pos_embed' in key:
                shape_old = loaded_state_dict[key_old].shape
                shape_new = model_state_dict[key].shape
                new_val = resize_pos_embed_1d(loaded_state_dict[key_old],
                                              shape_new)
                if shape_new == new_val.shape:
                    model_state_dict[key] = new_val
                    logger.info("[RESIZE] {} {} -> {} {}".format(
                        key_old, shape_old, key, shape_new))
                else:
                    logger.info("[WARNING]", "{} {} != {} {}, skip".format(
                        key_old, new_val.shape, key, shape_new))
            elif 'local_relative_position_bias_table' in key:
                shape_old = loaded_state_dict[key_old].shape
                shape_new = model_state_dict[key].shape
                new_val = resize_pos_embed_2d(loaded_state_dict[key_old],
                                              shape_new)
                if shape_new == new_val.shape:
                    model_state_dict[key] = new_val
                    logger.info("[RESIZE] {} {} -> {} {}".format(
                        key_old, shape_old, key, shape_new))
                else:
                    logger.info("[WARNING]", "{} {} != {} {}, skip".format(
                        key_old, new_val.shape, key, shape_new))
            else:
                # if layer weights does not match in size, skip this layer
                logger.info(
                    "SKIPPING LAYER {} because of size mis-match".format(key))
            continue
        model_state_dict[key] = loaded_state_dict[key_old]
        logger.info(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            )
        )


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def load_state_dict(model, loaded_state_dict):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict)

    # use strict loading
    model.load_state_dict(model_state_dict)
