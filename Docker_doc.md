## Code Step
#### Step 1. Build CUDA and PyTorch environment 
Default version
```bash
ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel  

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
```
#### Step.2 Install mmsegmentation
```bash
RUN conda clean --all
RUN pip install mmcv-full==1.3.0+torch1.6.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
RUN git clone https://github.com/open-mmlab/mmsegmentation.git /mmsegmentation
WORKDIR /mmsegmentation
RUN pip install -r requirements/readthedocs.txt \
	&& pip install --no-cache-dir -e .
```
#### Step.3 Install Swin Transformer from Github
```bash
RUN git clone https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation.git /usr/swin-seg
WORKDIR /usr/swin-seg
RUN pip install -r requirements.txt \
	&& pip install mmcv==1.3.0 \
	&& pip uninstall -y opencv-python \
	&& pip install opencv-python-headless \
```
#### Step.4 Download pre-train model and dataset Run sample code.
```shell
RUN mkdir pre_train \
	&& curl https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window8_256.pth -o pre_train/swinv2_tiny_patch4_window8_256.pth \
	&& tools/dist_train.sh configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py  1 --options model.pretrained=pre_train/swinv2_tiny_patch4_window8_256.pth
```

## Build Problem-Solving Note
#### <span style="color:#CF7A73; font-weight:bold" id="name"> ImportError: libGL.so.1: cannot open shared object file: No such file or directory [Solved]</span>
```bash
RUN pip uninstall opencv-python && \ 
	pip install opencv-python-headless
```
#### <span style="color:#CF7A73; font-weight:bold">pip3 install --upgrade -r requirements.txt error: these packages do not match the hashes from the requirements file</span> 

