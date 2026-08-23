"""MNIST 手写数字识别 —— 批次版训练测试脚本（基于 numpy_impl/network_batch）。

功能概览：
  1. 加载并解析本地 MNIST 训练/测试集（由 download_mnist.py 下载）。
  2. 展示数据集基本信息。
  3. 预处理：像素归一化 [0,1]，标签转 one-hot。
  4. 使用批次版 MLP（network_batch）按 batch 训练，输出每轮 loss 与验证准确率。
  5. 在测试集上输出平均 loss 与准确率。

本文件为新增，不覆盖 numpy_impl/test.py 骨架。

运行方式：
  python data/download_mnist.py      // 首次需先下载数据
  python numpy_impl/test_batch.py    // 从项目根运行
"""

import os
import struct
import sys

import numpy as np

# 将项目根加入 sys.path，使 `from numpy_impl.network_batch import ...` 可解析
PROJECT_ROOT = "c:/Users/Administrator/Desktop/Code/Proj/Ai/myMLP"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from numpy_impl.network_batch import network_batch

# ------------------------- 路径配置（与现有脚本一致） -------------------------
DATA_DIR = "c:/Users/Administrator/Desktop/Code/Proj/Ai/myMLP/data"
TRAIN_IMAGES = os.path.join(DATA_DIR, "train-images-idx3-ubyte")
TRAIN_LABELS = os.path.join(DATA_DIR, "train-labels-idx1-ubyte")
TEST_IMAGES = os.path.join(DATA_DIR, "t10k-images-idx3-ubyte")
TEST_LABELS = os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte")


# ------------------------- 1. 解析 IDX 格式 -------------------------
def load_idx_images(path):
    """解析 IDX3 图像文件，返回形状 (N, H, W) 的 uint8 数组。"""
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"图像文件魔数错误: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)


def load_idx_labels(path):
    """解析 IDX1 标签文件，返回形状 (N,) 的 uint8 数组。"""
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"标签文件魔数错误: {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8)


# ------------------------- 2. 预处理 -------------------------
def preprocess(x, y, num_classes=10):
    """归一化像素到 [0,1]，并将标签转为 one-hot。返回 (x_flat[N,784], y_onehot[N,10])。"""
    x_norm = x.reshape(x.shape[0], -1).astype(np.float32) / 255.0
    y_onehot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    y_onehot[np.arange(y.shape[0]), y] = 1.0
    return x_norm, y_onehot


# ------------------------- 3. 训练与测试（批次版） -------------------------
def main():
    # 1) 加载并解析数据
    print(">> 加载并解析 MNIST 数据 ...")
    x_train = load_idx_images(TRAIN_IMAGES)
    y_train = load_idx_labels(TRAIN_LABELS)
    x_test = load_idx_images(TEST_IMAGES)
    y_test = load_idx_labels(TEST_LABELS)
    print(f"train: {x_train.shape}, labels: {y_train.shape}")
    print(f"test : {x_test.shape}, labels: {y_test.shape}")

    # 2) 预处理
    print(">> 数据归一化 + one-hot ...")
    x_train, y_train = preprocess(x_train, y_train)
    x_test, y_test = preprocess(x_test, y_test)

    # 3) 从训练集中切出验证集（评估用），剩余用于训练
    val_size = 5000
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_tr, y_tr = x_train[val_size:], y_train[val_size:]
    print(f"train: {x_tr.shape}, val: {x_val.shape}, test: {x_test.shape}")

    # 4) 构建批次版神经网络并训练
    net = network_batch(input_size=784, hidden_dim=128,
                        num_hidden_layers=1, output_size=10)
    print(">> 开始批次训练（batch_size=32, lr=0.005, momentum=0.9）...")
    net.train(x_tr, y_tr, epochs=10, lr=0.005,
              batch_size=32, momentum=0.9,
              x_val=x_val, y_val=y_val, eval_every=1)

    # 5) 测试集评估
    test_loss, test_acc = net.evaluate(x_test, y_test)
    print("=" * 50)
    print(f"测试集 平均损失 = {test_loss:.6f}, 准确率 = {test_acc:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"数据文件缺失：{e}")
        print("请先运行：python data/download_mnist.py")
        sys.exit(1)