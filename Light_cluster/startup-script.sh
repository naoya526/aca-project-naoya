le ! ping -c 1 google.com &> /dev/null; do
  echo "Waiting for network..."
  sleep 5
done
B. パッケージロックの問題
bash
# apt-get実行前にロック解除を待機
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
  echo "Waiting for dpkg lock..."
  sleep 5
done
C. 権限の問題
GitHubからクローンす





#!/bin/bash

# ログ出力の設定
exec > >(tee -a /var/log/startup-script.log)
exec 2>&1

echo "=== Startup script started at $(date) ==="

# エラー時に停止
set -e

# 環境変数の設定
export DEBIAN_FRONTEND=noninteractive

echo "Updating package lists..."
sudo apt-get update -y

echo "Installing OpenMPI..."
sudo apt-get install -y openmpi-bin openmpi-common libopenmpi-dev

echo "Installing Git..."
sudo apt-get install -y git

echo "Cloning repository..."
# ホームディレクトリを明示的に指定
cd /home/debian
sudo -u debian git clone https://github.com/naoya526/aca-project-naoya

# クローンしたディレクトリの所有者を変更
sudo chown -R debian:debian /home/debian/aca-project-naoya

echo "=== Startup script completed successfully at $(date) ==="