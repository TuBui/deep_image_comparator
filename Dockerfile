FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && apt-get install -y \
    ca-certificates \
    git \
    wget \
    curl \
    bzip2 \
    libgtk2.0-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV HOME=/opt
ENV PATH=/opt/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.5 \
 && conda clean -ya

# CUDA 10.1-specific steps
RUN conda install -y -c pytorch \
    cudatoolkit=10.1 \
    pytorch==1.7.0 torchvision==0.8.0 \
    && conda clean -ya

# imagenet c
RUN git clone https://github.com/hendrycks/robustness.git && cd robustness/ImageNet-C/imagenet_c/ && pip install -e .

# imagemagik and opencv dependencies
RUN apt-get update -y && apt-get install -y \
    libmagickwand-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# pip packages
ADD ./requirements.txt ./
RUN pip install -r requirements.txt

RUN apt-get autoremove -y && apt-get autoclean -y
