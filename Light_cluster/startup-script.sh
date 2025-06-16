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
sudo apt update && sudo apt install -y \
    build-essential \
    libopenmpi-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    libjpeg-dev
pip install --upgrade pip
pip install \
    numpy \
    mpi4py \
    pillow \
    matplotlib \
    psutil

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