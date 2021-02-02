## Neural Motif Feature Extractor 
This file provides documentation for the neural motif algorithm implemented under 
relation_head class. 


### Classes Implemented:
- NeuralMotifFeatureExtractor under `roi_relation_feature_extractors.py`
    
    This class implements the main feature computation logic from 
    [Neural Motifs](https://github.com/rowanz/neural-motifs), 
    mainly covering section 4.2 and 4.3 of the paper.
- context_encoder under `neural_motif/context_encoder.py`

    This class implements the two LSTMs applied in Neural Motif, which 
    corresponds to object context and edge context. This class is initialized
    and called under NeuralMotifFeatureExtractor.
    
- roi_sorter under `neural_motif/roi_sorter.py`

    This class implements the sorting interface/algorithm for sorting 
    the proposals, based on a score metric with various definitions, as per
    Neural Motif paper. This class is initialized/called under context_encoder class.
    
    
- DecoderRNN under `neural_motif/decoder_rnn.py`

    This class implements the decoder RNN mentioned in section 4.2. It is 
    initialized/called under context_encoder class.
    
- NeuralMotifPredictor under `roi_relation_predictors.py`

    This class implements an interface for Eqn 7 of the Neural Motif paper. In short,
     a predictor given relation features.

- `word_vectors.py` this is not a class but it contains various helper functions
    that is used by Neural Motif. For instance, the word vector embedding that is 
    used for initializing object embeddings in object LSTM is implemented here.
    


### Datasets

There are two items that need to be prepared before running Neural Motif feature extractor.

- `datasets/visualgenome`
    This is a *.tsv format version of Danfei Xu's visual genome data split. Please ask Pengchuan 
    Zhang for the dataset access. For now a `*.zip` file containing the dataset is also provided.

- `datasets/glove` 
    This folder should contain glove database for word embedding look-up.
    The glove DB is used by `relation_head/neural_motif/word_vectors.py` to build 
    object class embeddings.

### Pretrained Detector Model

There is a necessity that you prepare a pretrained faster RCNN model, otherwise the base faster RCNN
model will have randomly initialized parameters thus your results will be adhoc and bad. The pretrained
 detector model should be located at `pretrained_model/faster_rcnn_ckpt.pth`.
 
This model and aforementioned datasets will be provided in a tar file, under:


    /media/data/t-chuw/neural_motif_datasets.tar.gz

You can access this folder from CCPIU04 machine. 

### Sample commands

- Train a relation learning network with Neural Motif algorithm. Please change `nproc_per_node` 
according to the number of GPUs you plan to use.


    python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --config-file configs/vg/relation_danfeiX_RES101_neural_motif.yaml


- Test.    


### Config settings 

There are several key variables/parameters that are special to Neural Motif head.

- ROI_RELATION_HEAD.NEURAL_MOTIF_ON
    
    This parameter controls the meta behavior of relation head. It has to be set as on
    in order for the NeuralMotif feature extractor to be active.
    
- ROI_RELATION_HEAD.MODE

    This controls the mode that Neural Motif uses. Currently `sgdet` is implemented.
    
- ROI_RELATION_HEAD.NEURAL_MOTIF.USE_TANH

    This controls whether to use tanh activation for relation feature output.

- ROI_RELATION_HEAD.NEURAL_MOTIF.ORDER

    This controls the sorting order for objects, before sending into LSTM.
    
- ROI_RELATION_HEAD.NEURAL_MOTIF.NUM_OBJS
    
    This controls the number of objects (output from the detector), that will be used
     in Neural Motif algorithm.
     
- Several path related parameters, please ensure correct files are in place:

    ROI_RELATION_HEAD.NEURAL_MOTIF.OBJ_CLASSES_FN : object class names for VG dataset.
    ROI_RELATION_HEAD.NEURAL_MOTIF.REL_CLASSES_FN : relation class names for VG dataset.
    
    ROI_RELATION_HEAD.NEURAL_MOTIF.GLOVE_PATH : glove db path for obj embeddings.


  
    
