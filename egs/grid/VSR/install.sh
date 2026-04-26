#!/usr/bin/env bash
set -euo pipefail

stage=1
ROOT_DIR=/data
ICEFALL_DIR="$ROOT_DIR/icefall"
AVHUBERT_DIR="$ROOT_DIR/av_hubert"

#
k2_wheel="k2-1.24.4.dev20241030+cuda12.1.torch2.4.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"

if [ $stage -le 1 ] ; then

  if  [ ! -f "$k2_wheel" ]; then
    echo "Here!"
    wget https://huggingface.co/csukuangfj/k2/resolve/main/ubuntu-cuda/1.24.4.dev20241029/$k2_wheel
  fi

  python -m pip install -r requirements.txt

  if [ ! -d "$ICEFALL_DIR" ]; then
    git clone https://github.com/k2-fsa/icefall "$ICEFALL_DIR"
  fi
  python -m pip install -r "$ICEFALL_DIR/requirements.txt"
fi

if [ $stage -le 2 ] ; then
  if [ ! -d "$AVHUBERT_DIR" ]; then
    git clone https://github.com/facebookresearch/av_hubert.git --depth=1 "$AVHUBERT_DIR"
  fi

  cd "$AVHUBERT_DIR"
  git submodule init
  git submodule update

  cd fairseq
  pip install --editable ./
fi

cd $ICEFALL_DIR/egs/grid/VSR
if [ $stage -le 3 ] ; then
  conda install -y -c conda-forge dlib==19.18.0
  python -m pip install -r grid-requirements.txt

  echo 'export PYTHONPATH=/data/git/icefall:$PYTHONPATH'
  echo 'Add the line above to your ~/.bashrc or conda activate hook'
fi