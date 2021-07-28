## Model Zoo and Baselines


### Openimages V5 Visual Relation Detection

TODO add yaml file link


All the following models are inferenced using unconstraint method, the detection part use X152FPN pretrained on OpenImages, COCO, Visual Genome and Object365 dataset.

model | recall@50 | wmAP(Triplet) | mAP(Triplet) | wmAP(Phrase) | mAP(Phrase) | Triplet proposal recall | Phrase proposal recall | model | config
-----------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:
IMP, no bias | 71.64 | 30.56 | 36.47 | 32.90 | 40.61 | 72.57 | 75.87 | [link](https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/sgg_model_zoo/oi_R152_imp_nobias.pth) | [link](sgg_configs/oi_vrd/R152FPN_imp_nobias_oi.yaml)
IMP, bias | 71.81 | 30.88 | 45.97 | 33.25 | 50.42 | 72.81 | 76.04 | [link](https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/sgg_model_zoo/oi_R152_imp_bias.pth) | [link](sgg_configs/oi_vrd/R152FPN_imp_bias_oi.yaml)
MSDN, no bias | 71.76 | 30.40 | 36.76 | 32.81 | 40.89 | 72.54 | 75.85 | [link](https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/sgg_model_zoo/oi_R152_msdn_nobias.pth) | [link](sgg_configs/oi_vrd/R152FPN_msdn_nobias_oi.yaml)
MSDN, bias | 71.48 | 30.22 | 34.49 | 32.58 | 38.71 | 72.45 | 75.62 | [link](https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/sgg_model_zoo/oi_R152_msdn_bias.pth) | [link](sgg_configs/oi_vrd/R152FPN_msdn_bias_oi.yaml)
Neural Motif, bias | 72.54 | 29.35 | 29.26 | 33.10 | 35.02 | 73.64 | 78.70 | [link](https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/sgg_model_zoo/oi_R152_nm.pth) | [link](sgg_configs/oi_vrd/R152FPN_motif_oi.yaml)
GRCNN, bias | 74.17 | 34.73 | 39.56 | 37.04 | 43.63 | 74.11 | 77.32 | [link](https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/sgg_model_zoo/oi_R152_grcnn.pth) | [link](sgg_configs/oi_vrd/R152FPN_grcnn_oi.yaml)
RelDN | 75.40 | 40.85 | 44.24 | 49.16 | 50.60 | 78.74 | 90.39 | [link](https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/sgg_model_zoo/oi_R152_reldn.pth) | [link](sgg_configs/oi_vrd/R152FPN_reldn_oi.yaml)


### Visual Genome

model | sgdet@20 | sgdet@50 | sgdet@100 | sgcls@20 | sgcls@50 | sgcls@100 | predcls@20 | predcls@50 | predcls@100 | model | config 
-----------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:
IMP, no bias | 19.8 | 27.5 | 33.0 | 28.0 | 33.4 | 35.1 | 44.9 | 54.8 | 57.8 | [link](https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/sgg_model_zoo/visualgenome/imp_usefpFalse_lr0.005_bsz4_featstep2/model_final.pth) | [link](sgg_configs/vg_vrd/rel_danfeiX_FPN50_imp.yaml)
IMP, bias | 21.7 | 29.3 | 34.5 | 29.2 | 33.9 | 35.3 | 48.8 | 57.6 | 59.9 | [link](https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/sgg_model_zoo/visualgenome/imp_usefpTrue_lr0.005_bsz4_featstep2/model_0120000.pth) | [link](sgg_configs/vg_vrd/rel_danfeiX_FPN50_imp.yaml)
MSDN, no bias | 21.0 | 28.3 | 33.5 | 28.2 | 33.4 | 35.0 | 46.0 | 55.0 | 57.7 | [link](https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/sgg_model_zoo/visualgenome/msdn_usefpFalse_lr0.005_bsz4_feaststep2/model_0100000.pth) | [link](sgg_configs/vg_vrd/rel_danfeiX_FPN50_msdn.yaml)
MSDN, bias | 22.4 | 30.0 | 35.3 | 29.7 | 34.4 | 35.9 | 51.2 | 59.6 | 61.6 | [link](https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/sgg_model_zoo/visualgenome/msdn_usefpTrue_lr0.005_bsz4_featstep2/model_0060000.pth) | [link](sgg_configs/vg_vrd/rel_danfeiX_FPN50_msdn.yaml)
Neural Motif, no bias | 21.0 | 28.6 | 33.8 | 29.2 | 34.1 | 35.5 | 51.0 | 60.2 | 62.3 | [link](https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/sgg_model_zoo/visualgenome/nm_usefpFalse_lr0.015_bsz8_objctx0_edgectx2_shareboxFalse/model_0035000.pth) | [link](sgg_configs/vg_vrd/rel_danfeiX_FPN50_nm.yaml)
Neural Motif, bias | 21.8 | 30.1 | 33.8 | 30.2 | 35.1 | 36.5 | 52.1 | 61.2 | 63.2 | [link](https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/sgg_model_zoo/visualgenome/nm_usefpTrue_lr0.015_bsz8_objctx0_edgectx2_shareboxFalse/model_final.pth) | [link](sgg_configs/vg_vrd/rel_danfeiX_FPN50_nm.yaml)
GRCNN, no bias | 20.5 | 27.4 | 35.7 | 27.1 | 31.5 | 32.9 | 42.5 | 50.7 | 53.3 | [link](https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/sgg_model_zoo/visualgenome/grcnn_usefpFalse_lr0.005_bsz4_featstep2_scorestep2/model_final.pth) | [link](sgg_configs/vg_vrd/rel_danfeiX_FPN50_grcnn.yaml) 
GRCNN, bias | 22.9 | 30.1 | 34.8 | 30.5 | 34.9 | 36.2 | 52.1 | 59.9 | 61.8 | [link](https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/sgg_model_zoo/visualgenome/grcnn_usefpTrue_lr0.005_bsz4_featstep2_scorestep2/model_0060000.pth) | [link](sgg_configs/vg_vrd/rel_danfeiX_FPN50_grcnn.yaml) 
RelDN | 24.0 | 32.4 | 37.8 | 31.9 | 35.7 | 36.6 | 54.0 | 60.9 | 62.5 | [link](https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/sgg_model_zoo/visualgenome/reldn_usefpTrue_lr0.005_bsz4/model_final.pth) | [link](sgg_configs/vg_vrd/rel_danfeiX_FPN50_reldn.yaml)