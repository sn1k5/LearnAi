"""MNIST 手写数字识别 —— 测试脚本（基于 numpy_impl 实现骨架）。

功能概览与 list_impl/test_mnist.py 一致，待 numpy_impl 各模块实现后填充。

运行方式：
  python data/download_mnist.py            # 首次需先下载数据
  python numpy_impl/test.py                # 从项目根运行
"""

import os
import struct
import sys

import numpy as np

# 将项目根加入 sys.path，使 `from numpy_impl.network import ...` 可解析
PROJECT_ROOT = "c:/Users/Administrator/Desktop/Code/Proj/Ai/myMLP"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from numpy_impl.network import network


# ------------------------- 路径配置（写死绝对路径） -------------------------
DATA_DIR = "c:/Users/Administrator/Desktop/Code/Proj/Ai/myMLP/data"
TRAIN_IMAGES = os.path.join(DATA_DIR, "train-images-idx3-ubyte")
TRAIN_LABELS = os.path.join(DATA_DIR, "train-labels-idx1-ubyte")
TEST_IMAGES = os.path.join(DATA_DIR, "t10k-images-idx3-ubyte")
TEST_LABELS = os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte")


# ------------------------- 1. 解析 IDX 格式 -------------------------
def load_idx_images(path):
    


def load_idx_labels(path):
    ...


# ------------------------- 2. 基本信息展示 -------------------------
def show_dataset_info(x, y):
    ...


# ------------------------- 3. 可视化 -------------------------
def visualize_samples(x, y, n=10, seed=42):
    ...


# ------------------------- 4. 预处理 -------------------------
def preprocess(x, y, num_classes=10):
    ...


# ------------------------- 5. 训练与测试 -------------------------
def main():
    ...


if __name__ == "__main__":
    main()
