# Minimal runtime for the GRID VSR demo (conformer_ctc2/demo.py).
#
# The heavy environment is NOT baked in: the host's conda env and the icefall
# repo are bind-mounted read-only by run.sh; this image only provides the
# system pieces the demo shells out to (ffmpeg) or links against (OpenCV's
# libGL/glib). GPU access comes from `docker run --gpus all` via the NVIDIA
# container toolkit -- the pip torch/k2 wheels bundle their own CUDA libs.
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# k2's _k2.so links libnvrtc/cuBLAS/etc. at load time; resolve them from the
# CUDA runtime wheels bundled in the (mounted) env rather than a host CUDA.
ENV NVLIBS=/data/miniconda3/envs/icefall-vsr/lib/python3.8/site-packages/nvidia
ENV PATH=/data/miniconda3/envs/icefall-vsr/bin:$PATH \
    PYTHONPATH=/data/icefall \
    LD_LIBRARY_PATH=$NVLIBS/cuda_nvrtc/lib:$NVLIBS/cublas/lib:$NVLIBS/cuda_runtime/lib:$NVLIBS/cudnn/lib:$NVLIBS/cufft/lib:$NVLIBS/curand/lib:$NVLIBS/cusolver/lib:$NVLIBS/cusparse/lib:$NVLIBS/nccl/lib:$NVLIBS/nvjitlink/lib:$NVLIBS/nvtx/lib \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_ANALYTICS_ENABLED=False \
    HOME=/tmp

WORKDIR /data/icefall/egs/grid/VSR
