# this installs the right pip and dependencies for the fresh python
conda install ipython h5py nltk joblib jupyter pandas scipy -y

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs>=0.1.8 cython matplotlib tqdm numpy>=1.19.5

#conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch -y
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=9.2 -c pytorch -y

conda install -c conda-forge timm einops -y

# install pycocotools
conda install -c conda-forge pycocotools -y

# install cityscapesScripts
python -m pip install cityscapesscripts

# install opencv-python
conda install -c conda-forge opencv -y

