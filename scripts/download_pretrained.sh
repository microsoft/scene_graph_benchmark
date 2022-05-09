# Download pretrained models

export MAIN_DIR=$PWD

mkdir -p checkpoints

# Causal-TDE: Unbiased Neural Motifs
cd checkpoints
mkdir causal_tde
cd causal_tde
#wget https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/sgg_model_zoo/visualgenome/nm_usefpTrue_lr0.015_bsz8_objctx0_edgectx2_shareboxFalse/model_final.pth -O "model_final.pth"
wget https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/sgg_model_zoo/visualgenome/nm_usefpFalse_lr0.015_bsz8_objctx0_edgectx2_shareboxFalse/model_0035000.pth -O "model_0035000.pth"
#wget https://raw.githubusercontent.com/microsoft/scene_graph_benchmark/main/sgg_configs/vg_vrd/rel_danfeiX_FPN50_nm.yaml -O "rel_danfeiX_FPN50_nm.yaml"

cd $MAIN_DIR
