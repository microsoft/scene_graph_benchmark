## Model Zoo and Baselines


### Openimages V5 Visual Relation Detection

TODO add yaml file link


All the following models are inferenced using unconstraint method, the detection part use X152FPN pretrained on OpenImages, COCO, Visual Genome and Object365 dataset.

model | recall@50 | wmAP(Triplet) | mAP(Triplet) | wmAP(Phrase) | mAP(Phrase) | Triplet proposal recall | Phrase proposal recall | model | config
-----------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:
IMP, no bias | 71.64 | 30.56 | 36.47 | 32.90 | 40.61 | 72.57 | 75.87 | [link]() | [link]()
IMP, bias | 71.81 | 30.88 | 45.97 | 33.25 | 50.42 | 72.81 | 76.04 | [link]() | [link]()
MSDN, no bias | 71.76 | 30.40 | 36.76 | 32.81 | 40.89 | 72.54 | 75.85 | [link]() | [link]()
MSDN, bias | 71.48 | 30.22 | 34.49 | 32.58 | 38.71 | 72.45 | 75.62 | [link]() | [link]()
Neural Motif, bias | 72.54 | 29.35 | 29.26 | 33.10 | 35.02 | 73.64 | 78.70 | [link]() | [link]()
GRCNN, bias | 74.17 | 34.73 | 39.56 | 37.04 | 43.63 | 74.11 | 77.32 | [link]() | [link]()
RelDN | 75.40 | 40.85 | 44.24 | 49.16 | 50.60 | 78.74 | 90.39 | [link]() | [link]()


### Visual Genome

model | sgdet@20 | sgdet@50 | sgdet@100 | sgcls@20 | sgcls@50 | sgcls@100 | predcls@20 | predcls@50 | predcls@100 | model | config 
-----------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:
IMP | 19.8 | 27.5 | 33.0 |  |  |  |  |  |  | [link](https://penzhanwu2.blob.core.windows.net/phillytools/vg_jwy/R50FPN_imp_no_pre_computedcontrastive_loss.use_flag_false_seperate_so_feature_extractor_false_use_bias_false_imp_feature_update_step_2/model_final.pth) | [link]()
IMP-bias | 21.7 | 29.3 | 34.5 |  |  |  |  |  |  | [link](https://penzhanwu2.blob.core.windows.net/phillytools/vg_jwy/R50FPN_imp_no_pre_computedcontrastive_loss.use_flag_false_seperate_so_feature_extractor_false_use_bias_true/model_0120000.pth) | [link]()
MSDN | 20.6 | 28.1 | 33.3 |  | |  |  |  |  | [link](https://penzhanwu2.blob.core.windows.net/phillytools/vg_jwy/R50FPN_msdn_no_pre_computedcontrastive_loss.use_flag_false_seperate_so_feature_extractor_false_use_bias_false_msdn_feature_update_step_2/model_0100000.pth) | [link]()
MSDN-bias | 22.4 | 30.0 | 35.3 |  |  |  | |  | | [link](https://penzhanwu2.blob.core.windows.net/phillytools/vg_jwy/R50FPN_msdn_no_pre_computedcontrastive_loss.use_flag_false_seperate_so_feature_extractor_false_use_bias_true/model_0060000.pth) | [link]()
Neural Motif |  |  |  |  |  |  |  |  |  | [link]() | [link]()
Neural Motif-bias |  |  | |  | |  |  |  | | [link]() | [link]()
GRCNN | 20.5 | 27.4 | 31.9 |  |  |  |  |  |  | [link](https://penzhanwu2.blob.core.windows.net/phillytools/vg_jwy/R50FPN_grcnn_no_pre_computedcontrastive_loss.use_flag_false_seperate_so_feature_extractor_false_use_relpn_true_use_bias_false_grcnn_feature_update_step_2_grcnn_score_update_step_2/model_final.pth) | [link]() 
GRCNN-bias | 22.9 | 30.1 | 34.8 |  |  |  |  |  |  | [link](https://penzhanwu2.blob.core.windows.net/phillytools/vg_jwy/R50FPN_grcnn_no_pre_computedcontrastive_loss.use_flag_false_seperate_so_feature_extractor_false_use_relpn_true_use_bias_true_grcnn_feature_update_step_2_grcnn_score_update_step_2/model_0060000.pth) | [link]() 
RelDN | 24.0 | 32.4 | 37.8 | - | - | - | - | - | - | [link](https://penzhanwu2.blob.core.windows.net/phillytools/vg_jwy/R50FPN_vrd_no_pre_computedcontrastive_loss.use_flag_true_seperate_so_feature_extractor_true_use_bias_true/model_final.pth) | [link]()

