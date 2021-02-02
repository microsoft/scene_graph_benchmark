# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import errno
import json
import yaml
import shutil
import logging
import os
import re
import numpy as np
import torch
import random
from .comm import is_main_process


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_labels(dataset_list, output_dir):
    if is_main_process():
        logger = logging.getLogger(__name__)

        ids_to_labels = {}
        for dataset in dataset_list:
            if hasattr(dataset, 'categories'):
                ids_to_labels.update(dataset.categories)
            else:
                logger.warning("Dataset [{}] has no categories attribute, labels.json file won't be created".format(
                    dataset.__class__.__name__))

        if ids_to_labels:
            labels_file = os.path.join(output_dir, 'labels.json')
            logger.info("Saving labels mapping into {}".format(labels_file))
            with open(labels_file, 'w') as f:
                json.dump(ids_to_labels, f, indent=2)


def save_config(cfg, path):
    if is_main_process():
        with open(path, 'w') as f:
            f.write(cfg.dump())


def config_iteration(output_dir, max_iter):
    save_file = os.path.join(output_dir, 'last_checkpoint')
    iteration = -1
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            fname = f.read().strip()
        model_name = os.path.basename(fname)
        model_path = os.path.dirname(fname)
        if model_name.startswith('model_') and len(model_name) == 17:
            iteration = int(model_name[-11:-4])
        elif model_name == "model_final":
            iteration = max_iter
        elif model_path.startswith('checkpoint-') and len(model_path) == 18:
            iteration = int(model_path.split('-')[-1])
    return iteration


def get_matching_parameters(model, regexp, none_on_empty=True):
    """Returns parameters matching regular expression"""
    if not regexp:
        if none_on_empty:
            return {}
        else:
            return dict(model.named_parameters())
    compiled_pattern = re.compile(regexp)
    params = {}
    for weight_name, weight in model.named_parameters():
        if compiled_pattern.match(weight_name):
            params[weight_name] = weight
    return params


def freeze_weights(model, regexp):
    """Freeze weights based on regular expression."""
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    for weight_name, weight in get_matching_parameters(model, regexp).items():
        weight.requires_grad = False
        logger.info("Disabled training of {}".format(weight_name))


def delete_tsv_files(tsvs):
    for t in tsvs:
        if os.path.isfile(t):
            try_delete(t)
        line = os.path.splitext(t)[0] + '.lineidx'
        if os.path.isfile(line):
            try_delete(line)


def concat_files(ins, out):
    mkdir(os.path.dirname(out))
    out_tmp = out + '.tmp'
    with open(out_tmp, 'wb') as fp_out:
        for i, f in enumerate(ins):
            logging.info('concating {}/{} - {}'.format(i, len(ins), f))
            with open(f, 'rb') as fp_in:
                shutil.copyfileobj(fp_in, fp_out, 1024*1024*10)
    os.rename(out_tmp, out)


def concat_tsv_files(tsvs, out_tsv):
    concat_files(tsvs, out_tsv)
    sizes = [os.stat(t).st_size for t in tsvs]
    sizes = np.cumsum(sizes)
    all_idx = []
    for i, t in enumerate(tsvs):
        for idx in load_list_file(os.path.splitext(t)[0] + '.lineidx'):
            if i == 0:
                all_idx.append(idx)
            else:
                all_idx.append(str(int(idx) + sizes[i - 1]))
    with open(os.path.splitext(out_tsv)[0] + '.lineidx', 'w') as f:
        f.write('\n'.join(all_idx))


def load_list_file(fname):
    with open(fname, 'r') as fp:
        lines = fp.readlines()
    result = [line.strip() for line in lines]
    if len(result) > 0 and result[-1] == '':
        result = result[:-1]
    return result


def try_once(func):
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.info('ignore error \n{}'.format(str(e)))
    return func_wrapper


@try_once
def try_delete(f):
    os.remove(f)


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def write_to_yaml_file(context, file_name):
    with open(file_name, 'w') as fp:
        yaml.dump(context, fp, encoding='utf-8')


def load_from_yaml_file(yaml_file):
    with open(yaml_file, 'r') as fp:
        return yaml.load(fp, Loader=yaml.CLoader)
