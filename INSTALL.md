## Installation

### Requirements:
- PyTorch 1.7
- torchvision
- cocoapi
- yacs>=0.1.8
- numpy>=1.19.5
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA >= 10.1


### Option 1: Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name sg_benchmark python=3.7 -y
conda activate sg_benchmark

# this installs the right pip and dependencies for the fresh python
conda install ipython h5py nltk joblib jupyter pandas scipy

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs>=0.1.8 cython matplotlib tqdm opencv-python numpy>=1.19.5

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
conda install -c conda-forge timm einops

# install pycocotools
conda install -c conda-forge pycocotools

# install cityscapesScripts
python -m pip install cityscapesscripts

# install Scene Graph Detection
git clone https://github.com/microsoft/scene_graph_benchmark
cd scene_graph_benchmark

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop


```
### Option 2: Docker Image (Requires CUDA, Linux only)

Build image with defaults (`CUDA=10.1`, `CUDNN=7`, `FORCE_CUDA=1`):

    nvidia-docker build -t scene_graph_benchmark docker/

Build image with FORCE_CUDA disabled:

    nvidia-docker build -t scene_graph_benchmark --build-arg FORCE_CUDA=0 docker/

Build and run image with built-in jupyter notebook(note that the password is used to log in jupyter notebook):

    nvidia-docker build -t scene_graph_benchmark-jupyter docker/docker-jupyter/
    nvidia-docker run -td -p 8888:8888 -e PASSWORD=<password> -v <host-dir>:<container-dir> scene_graph_benchmark-jupyter
