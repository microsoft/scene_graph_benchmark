ARG CUDA="10.1"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libyaml-dev vim zsh wget htop tmux

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.7 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name py37 python=3.7 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py37
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython h5py nltk joblib jupyter pandas scipy
RUN pip install requests ninja cython yacs>=0.1.8 numpy>=1.19.5 cython matplotlib opencv-python tqdm \
 protobuf tensorboardx pymongo sklearn boto3 scikit-image cityscapesscripts
RUN pip install azureml-defaults>=1.0.45 azureml.core inference-schema
RUN pip --no-cache-dir install --force-reinstall -I pyyaml

RUN python -m nltk.downloader punkt

# Install latest PyTorch 1.7.1
ARG CUDA
RUN conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch \
 && conda clean -ya
RUN conda install -y -c conda-forge timm einops

# install pycocotools
# RUN git clone https://github.com/cocodataset/cocoapi.git \
#  && cd cocoapi/PythonAPI \
#  && python setup.py build_ext install
RUN conda install -y -c conda-forge pycocotools

# install cityscapesScripts
RUN python -m pip install cityscapesscripts

# install PyTorch Detection
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}

RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
RUN echo """export ZSH="/root/.oh-my-zsh"\nZSH_THEME="gentoo"\nplugins=(git)\nsource \$ZSH/oh-my-zsh.sh""" > /root/.zshrc
RUN echo """syntax on\nfiletype indent on\nset autoindent\nset number\ncolorscheme desert""" > /root/.vimrc

CMD [ "zsh" ]

# RUN git clone https://github.com/microsoft/scene_graph_benchmark.git \
#  && cd scene_graph_benchmark \
#  && python setup.py build develop

# WORKDIR /scene_graph_benchmark
