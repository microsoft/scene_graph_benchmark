#!/bin/bash

set -e # exit on first error

while getopts c:f:t: flag
do
    case "${flag}" in
        c) clip=${OPTARG};;
        f) frame=${OPTARG};;
        t) threshold=${OPTARG};;
    esac
done

python --version

python tools/demo/clip2frame.py --path $clip --keyframe $frame
python tools/demo/generate_sg.py --config_file checkpoints/causal_tde/rel_danfeiX_FPN50_nm.yaml --img_file ${clip}_frame_$frame.jpg --visualize_attr --visualize_relation --filtering_trs $threshold  MODEL.ROI_RELATION_HEAD.DETECTOR_PRE_CALCULATED False DATASETS.LABELMAP_FILE "VG-SGG-dicts-danfeiX-clipped.json" DATA_DIR data/VG MODEL.ATTRIBUTE_ON True MODEL.RELATION_ON True TEST.OUTPUT_RELATION_FEATURE True MODEL.ROI_RELATION_HEAD.USE_BIAS True MODEL.ROI_RELATION_HEAD.FILTER_NON_OVERLAP True MODEL.ROI_HEADS.DETECTIONS_PER_IMG 64 MODEL.ROI_RELATION_HEAD.SHARE_BOX_FEATURE_EXTRACTOR False MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.OBJ_LSTM_NUM_LAYERS 0 MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.EDGE_LSTM_NUM_LAYERS 2 TEST.IMS_PER_BATCH 2
python tools/demo/demo_image.py --config_file checkpoints/causal_tde/rel_danfeiX_FPN50_nm.yaml --img_file ${clip}_frame_$frame.jpg --visualize_attr --visualize_relation --filtering_trs $threshold MODEL.ROI_RELATION_HEAD.DETECTOR_PRE_CALCULATED False DATASETS.LABELMAP_FILE "VG-SGG-dicts-danfeiX-clipped.json" DATA_DIR data/VG MODEL.ATTRIBUTE_ON True MODEL.RELATION_ON True TEST.OUTPUT_RELATION_FEATURE True MODEL.ROI_RELATION_HEAD.USE_BIAS True MODEL.ROI_RELATION_HEAD.FILTER_NON_OVERLAP True MODEL.ROI_HEADS.DETECTIONS_PER_IMG 64 MODEL.ROI_RELATION_HEAD.SHARE_BOX_FEATURE_EXTRACTOR False MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.OBJ_LSTM_NUM_LAYERS 0 MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.EDGE_LSTM_NUM_LAYERS 2 TEST.IMS_PER_BATCH 2
