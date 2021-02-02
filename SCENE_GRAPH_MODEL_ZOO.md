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
IMP | 10.5 | 13.8 | 16.1 | 10.5 | 13.8 | 16.1 | 10.5 | 13.8 | 16.1 | [link]() | [link]()
IMP-bias | 10.5 | 13.8 | 16.1 | 10.5 | 13.8 | 16.1 | 10.5 | 13.8 | 16.1 | [link]() | [link]()
MSDN | 10.5 | 13.8 | 16.1 | 10.5 | 13.8 | 16.1 | 10.5 | 13.8 | 16.1 | [link]() | [link]()
MSDN-bias | 10.5 | 13.8 | 16.1 | 10.5 | 13.8 | 16.1 | 10.5 | 13.8 | 16.1 | [link]() | [link]()
Neural Motif | 10.5 | 13.8 | 16.1 | 10.5 | 13.8 | 16.1 | 10.5 | 13.8 | 16.1 | [link]() | [link]()
Neural Motif-bias | 10.5 | 13.8 | 16.1 | 10.5 | 13.8 | 16.1 | 10.5 | 13.8 | 16.1 | [link]() | [link]()
GRCNN | 10.5 | 13.8 | 16.1 | 10.5 | 13.8 | 16.1 | 10.5 | 13.8 | 16.1 | [link]() | [link]() 
GRCNN-bias | 10.5 | 13.8 | 16.1 | 10.5 | 13.8 | 16.1 | 10.5 | 13.8 | 16.1 | [link]() | [link]() 
RelDN | 10.5 | 13.8 | 16.1 | 10.5 | 13.8 | 16.1 | 10.5 | 13.8 | 16.1 | [link]() | [link]()
