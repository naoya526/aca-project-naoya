#!/bin/bash

# デバッグ用：全てのコマンドと出力をログに記録
exec > >(tee -a /tmp/startup-debug.log)
exec 2>&1
set -x

echo "=== DEBUG: Startup script started at $(date) ==="
echo "Current user: $(whoami)"
echo "Current directory: $(pwd)"
echo "PATH: $PATH"

# ネットワーク接続確認
echo "Testing network connectivity..."
if ping -c 1 8.8.8.8; then
    echo "Network OK"
else
    echo "Network FAILED"
    exit 1
fi

# パッケージマネージャの状態確認
echo "Checking package manager..."
sudo apt-get update -y || { echo "apt-get update failed"; exit 1; }

# OpenMPIインストール
echo "Installing OpenMPI..."
sudo apt-get install -y openmpi-bin openmpi-common libopenmpi-dev || { echo "OpenMPI install failed"; exit 1; }

# Gitインストール
echo "Installing Git..."
sudo apt-get install -y git || { echo "Git install failed"; exit 1; }

# Python
echo "Installing Python Library..."
# Ubuntu パッケージを更新
sudo apt update -y

# Python3 と pip3 がインストールされていない場合に備えて
sudo apt install -y python3 python3-pip python3-venv

# Python パッケージ依存を apt でインストール
sudo apt install -y \
  python3-numpy \
  python3-mpi4py \
  libopenmpi-dev \
  openmpi-bin \
  python3-pil \
  python3-matplotlib \
  python3-psutil

# 動作確認
echo "インストール済みパッケージのバージョン確認:"
python3 -c "
import ctypes
import numpy as np
from mpi4py import MPI
import time
from PIL import Image
from matplotlib import pyplot as plt
import psutil
import os
import sys

print('すべてのパッケージが正常にインポートされました！')
"

sudo apt update
sudo apt install linux-tools-common linux-tools-$(uname -r)


# リポジトリクローン
echo "Cloning repository..."
cd /home/debian
if sudo -u debian git clone https://github.com/naoya526/aca-project-naoya; then
    echo "Git clone successful"
    sudo chown -R debian:debian /home/debian/aca-project-naoya
    ls -la /home/debian/
else
    echo "Git clone failed"
    exit 1
fi

echo "=== DEBUG: Startup script completed at $(date) ==="