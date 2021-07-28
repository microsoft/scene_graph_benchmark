# Scene Graph Benchmark in PyTorch 1.7

**This project is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)**

![alt text](demo/R152FPN_demo.png "from https://storage.googleapis.com/openimages/web/index.html")


## Highlights
- **Upgrad to pytorch 1.7**
- **Multi-GPU training and inference**
- **Batched inference:** can perform inference using multiple images per batch per GPU.
- **Fast and flexible tsv dataset format**
- **Remove FasterRCNN detector dependency:** during relation head training, can plugin bounding boxes from any detector.
- Provides pre-trained models for different scene graph detection algorithms ([IMP](https://arxiv.org/pdf/1701.02426.pdf), [MSDN](http://cvboy.com/publication/iccv2017_msdn/), [GRCNN](https://arxiv.org/pdf/1808.00191.pdf), [Neural Motif](https://arxiv.org/pdf/1711.06640.pdf), [RelDN](https://arxiv.org/pdf/1903.02728.pdf)).
- Provides bounding box level and relation level feature extraction functionalities
- Provides large detector backbones (ResNxt152)


## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.


## Model Zoo and Baselines

Pre-trained models can be found in [SCENE_GRAPH_MODEL_ZOO.md](SCENE_GRAPH_MODEL_ZOO.md)

## Visualization and Demo
We provide a helper class to simplify writing inference pipelines using pre-trained models (Currently only support objects and attributes).
Here is how we would do it. Run the following commands:
```bash
# visualize VinVL object detection
# pretrained models at https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth
# the associated labelmap at https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/VG-SGG-dicts-vgoi6-clipped.json
python tools/demo/demo_image.py --config_file sgg_configs/vgattr/vinvl_x152c4.yaml --img_file demo/woman_fish.jpg --save_file output/woman_fish_x152c4.obj.jpg MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth MODEL.ROI_HEADS.NMS_FILTER 1 MODEL.ROI_HEADS.SCORE_THRESH 0.2 TEST.IGNORE_BOX_REGRESSION False

# visualize VinVL object-attribute detection
# pretrained models at https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth
# the associated labelmap at https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/VG-SGG-dicts-vgoi6-clipped.json
python tools/demo/demo_image.py --config_file sgg_configs/vgattr/vinvl_x152c4.yaml --img_file demo/woman_fish.jpg --save_file output/woman_fish_x152c4.attr.jpg --visualize_attr MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth MODEL.ROI_HEADS.NMS_FILTER 1 MODEL.ROI_HEADS.SCORE_THRESH 0.2 TEST.IGNORE_BOX_REGRESSION False

# visualize OpenImage scene graph generation by RelDN
# pretrained models at https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/sgg_model_zoo/sgg_oi_vrd_model_zoo/RX152FPN_reldn_oi_best.pth
python tools/demo/demo_image.py --config_file sgg_configs/vrd/R152FPN_vrd_reldn.yaml --img_file demo/1024px-Gen_Robert_E_Lee_on_Traveler_at_Gettysburg_Pa.jpg --save_file output/1024px-Gen_Robert_E_Lee_on_Traveler_at_Gettysburg_Pa.reldn_relation.jpg --visualize_relation MODEL.ROI_RELATION_HEAD.DETECTOR_PRE_CALCULATED False

# visualize Visual Genome scene graph generation by neural motif
python tools/demo/demo_image.py --config_file sgg_configs/vg_vrd/rel_danfeiX_FPN50_nm.yaml --img_file demo/1024px-Gen_Robert_E_Lee_on_Traveler_at_Gettysburg_Pa.jpg --save_file demo/1024px-Gen_Robert_E_Lee_on_Traveler_at_Gettysburg_Pa_vgnm.jpg --visualize_relation MODEL.ROI_RELATION_HEAD.DETECTOR_PRE_CALCULATED False DATASETS.LABELMAP_FILE "visualgenome/VG-SGG-dicts-danfeiX-clipped.json" DATA_DIR /home/penzhan/GitHub/maskrcnn-benchmark-1/datasets1 MODEL.ROI_RELATION_HEAD.USE_BIAS True MODEL.ROI_RELATION_HEAD.FILTER_NON_OVERLAP True MODEL.ROI_HEADS.DETECTIONS_PER_IMG 64 MODEL.ROI_RELATION_HEAD.SHARE_BOX_FEATURE_EXTRACTOR False MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.OBJ_LSTM_NUM_LAYERS 0 MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.EDGE_LSTM_NUM_LAYERS 2 TEST.IMS_PER_BATCH 2
```

## Perform training

For the following examples to work, you need to first install this repo.

You will also need to download the dataset. Datasets can be downloaded by [azcopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10) with following command:
```bash
path/to/azcopy copy 'https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/datasets/TASK_NAME' <target folder> --recursive
```
`TASK_NAME` could be `visualgenome`, `openimages_v5c`.

We recommend to symlink the path to the dataset to `datasets/` as follows

```bash
# symlink the dataset
cd ~/github/maskrcnn-benchmark
mkdir -p datasets/openimages_v5c/
ln -s /vrd datasets/openimages_v5c/vrd
```

You can also prepare your own datasets.

Follow tsv dataset creation instructions [tools/mini_tsv/README.md](tools/mini_tsv/README.md)


### Single GPU training

```bash
python tools/train_sg_net.py --config-file "/path/to/config/file.yaml"
```
This should work out of the box and is very similar to what we should do for multi-GPU training.
But the drawback is that it will use much more GPU memory. The reason is that we set in the configuration files a global batch size that is divided over the number of GPUs. So if we only have a single GPU, this means that the batch size for that GPU will be 4x larger, which might lead to out-of-memory errors.


### Multi-GPU training
We use internally `torch.distributed.launch` in order to launch
multi-gpu training. This utility function from PyTorch spawns as many
Python processes as the number of GPUs we want to use, and each Python
process will only use a single GPU.

```bash
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_sg_net.py --config-file "path/to/config/file.yaml" 
```


## Evaluation
You can test your model directly on single or multiple gpus. 
To evaluate relations, one needs to output "relation_scores_all" in the TSV_SAVE_SUBSET.
Here are a few example command line for evaluating on 4 GPUS:
```bash
export NGPUS=4

python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file CONFIG_FILE_PATH 

# vg IMP evaluation
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file sgg_configs/vg_vrd/rel_danfeiX_FPN50_imp.yaml

# vg MSDN evaluation
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file sgg_configs/vg_vrd/rel_danfeiX_FPN50_msdn.yaml

# vg neural motif evaluation
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file sgg_configs/vg_vrd/rel_danfeiX_FPN50_nm.yaml

# vg GRCNN evaluation
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file sgg_configs/vg_vrd/rel_danfeiX_FPN50_grcnn.yaml

# vg RelDN evaluation
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file sgg_configs/vg_vrd/rel_danfeiX_FPN50_reldn.yaml

# oi IMP evaluation
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file sgg_configs/oi_vrd/R152FPN_imp_bias_oi.yaml

# oi MSDN evaluation
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file sgg_configs/oi_vrd/R152FPN_msdn_bias_oi.yaml

# oi neural motif evaluation
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file sgg_configs/oi_vrd/R152FPN_motif_oi.yaml

# oi GRCNN evaluation
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file sgg_configs/oi_vrd/R152FPN_grcnn_oi.yaml

# oi RelDN evaluation
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file sgg_configs/vrd/R152FPN_vrd_reldn.yaml
```

To evaluate in sgcls mode:
```bash
export NGPUS=4

python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file CONFIG_FILE_PATH MODEL.ROI_BOX_HEAD.FORCE_BOXES True MODEL.ROI_RELATION_HEAD.MODE "sgcls"
```

To evaluate in predcls mode:
```bash
export NGPUS=4

python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file CONFIG_FILE_PATH MODEL.ROI_RELATION_HEAD.MODE "predcls"
```

To evaluate with ground truth bbox and ground truth pairs:
```bash
export NGPUS=4

python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file CONFIG_FILE_PATH MODEL.ROI_RELATION_HEAD.FORCE_RELATIONS True
```


## Abstractions
For more information on some of the main abstractions in our implementation, see [ABSTRACTIONS.md](ABSTRACTIONS.md).

## Adding your own dataset

This implementation adds support for TSV style datasets.
But adding support for training on a new dataset can be done as follows:

```python
from maskrcnn_benchmark.data.datasets.relation_tsv import RelationTSVDataset

class MyDataset(RelationTSVDataset):
    def __init__(self, yaml_file, extra_fields=(), transforms=None,
            is_load_label=True, **kwargs):

        super(MyDataset, self).__init__(yaml_file, extra_fields, transforms, is_load_label, **kwargs)
    
    def your_own_function(self, idx, call=False):
        # you can overwrite function or add your own functions this way
        pass
```
That's it. You can also add extra fields to the boxlist, such as segmentation masks
(using `structures.segmentation_mask.SegmentationMask`), or even your own instance type.

For a full example of how the `VGTSVDataset` is implemented, check [`maskrcnn_benchmark/data/datasets/vg_tsv.py`](maskrcnn_benchmark/data/datasets/vg_tsv.py).

Once you have created your dataset, it needs to be added in a couple of places:
- [`maskrcnn_benchmark/data/datasets/__init__.py`](maskrcnn_benchmark/data/datasets/__init__.py): add it to `__all__`
- [`maskrcnn_benchmark/data/datasets/utils/config_args.py`](maskrcnn_benchmark/data/datasets/utils/config_args.py): add it's name as an option to `tsv_dataset_name`


### Adding your own evaluation
To enable your dataset for testing, add a corresponding if statement in [`maskrcnn_benchmark/data/datasets/evaluation/__init__.py`](maskrcnn_benchmark/data/datasets/evaluation/__init__.py):
```python
if isinstance(dataset, datasets.MyDataset):
        return your_evaluation(**args)
```


## VinVL Feature extraction

The output feature will be encoded as base64
```bash
# extract vision features with VinVL object-attribute detection model
# pretrained models at https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth
# the associated labelmap at https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/VG-SGG-dicts-vgoi6-clipped.json
python tools/test_sg_net.py --config-file sgg_configs/vgattr/vinvl_x152c4.yaml TEST.IMS_PER_BATCH 2 MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth MODEL.ROI_HEADS.NMS_FILTER 1 MODEL.ROI_HEADS.SCORE_THRESH 0.2 DATA_DIR "../maskrcnn-benchmark-1/datasets1" TEST.IGNORE_BOX_REGRESSION True MODEL.ATTRIBUTE_ON True
```
To extract relation features (union bounding box's feature), in yaml file, set `TEST.OUTPUT_RELATION_FEATURE` to  `True`, add `relation_feature` in `TEST.TSV_SAVE_SUBSET`. 

To extract bounding box features, in yaml file, set `TEST.OUTPUT_FEATURE` to  `True`, add `feature` in `TEST.TSV_SAVE_SUBSET`.


## Troubleshooting
If you have issues running or compiling this code, we have compiled a list of common issues in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md). If your issue is not present there, please feel
free to open a new issue.

## Citations
Please consider citing this project in your publications if it helps your research. The following is a BibTeX reference. The BibTeX entry requires the `url` LaTeX package.
```
@misc{han2021image,
      title={Image Scene Graph Generation (SGG) Benchmark}, 
      author={Xiaotian Han and Jianwei Yang and Houdong Hu and Lei Zhang and Jianfeng Gao and Pengchuan Zhang},
      year={2021},
      eprint={2107.12604},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```

  
## License

maskrcnn-benchmark is released under the MIT license. See [LICENSE](LICENSE) for additional details.

## Acknowledgement
