# FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04
# RUN apt-get -y update
# RUN apt install -y software-properties-common
# RUN add-apt-repository -y ppa:deadsnakes/ppa
# RUN apt install -y python3.8 python3-pip git curl
# RUN alias python=python3.8 && alias pip=pip3

# ARG PYTORCH="1.6.0"
# ARG CUDA="10.1"
# ARG CUDNN="7"

# FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
# ENV CUDA_HOME "/usr/local/cuda"
# RUN git clone https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation.git /usr/swin-seg
# WORKDIR /usr/swin-seg
# RUN mkdir pre_train && curl https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window8_256.pth -o pre_train/swinv2_tiny_patch4_window8_256.pth
# RUN pip3 install --upgrade pip
# RUN pip3 install --upgrade --no-cache-dir mmcv-full==1.3.0
# RUN pip3 install --upgrade -r requirements.txt
# RUN pip3 install --no-cache-dir torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
# RUN tools/dist_train.sh configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py  1 --options model.pretrained=pre_train/swinv2_tiny_patch4_window8_256.pth




ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install mmsegmentation
RUN conda clean --all
RUN pip install mmcv-full==1.3.0+torch1.6.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
RUN git clone https://github.com/open-mmlab/mmsegmentation.git /mmsegmentation
WORKDIR /mmsegmentation
RUN pip install -r requirements/readthedocs.txt \
    && pip install --no-cache-dir -e .

# Clone swin-seg git
RUN git clone https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation.git /usr/swin-seg 
WORKDIR /usr/swin-seg
RUN mkdir pre_train && curl https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window8_256.pth -o pre_train/swinv2_tiny_patch4_window8_256.pth \
    && pip install -r requirements.txt \
    && pip install mmcv==1.3.0 \
    && pip uninstall -y opencv-python \
    && pip install opencv-python-headless \
# Cleanup
    && apt-get clean \
    && rm -rf $HOME/.cache/pip
