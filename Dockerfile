ARG CUDA="10.1"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics

#RUN apt-get autoclean

RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libyaml-dev nano wget \
 && apt-get install -y ffmpeg libgl1-mesa-glx

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name kern_nemesis python=3.7 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=kern_nemesis
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda init bash
RUN conda install -y ipython h5py nltk joblib jupyter pandas scipy
RUN pip install requests ninja cython yacs>=0.1.8 numpy>=1.19.5 cython matplotlib opencv-python \
 protobuf tensorboardx pymongo sklearn boto3 scikit-image cityscapesscripts
RUN pip install azureml-defaults>=1.0.45 azureml.core inference-schema opencv-python timm einops 
RUN pip --no-cache-dir install --force-reinstall -I pyyaml

# Install latest PyTorch 1.7.1
ARG CUDA
#RUN conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=9.2 -c pytorch \
 #&& conda clean -ya

RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

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

WORKDIR /kern_nemesis

