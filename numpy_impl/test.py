"""MNIST 手写数字识别 —— 测试脚本（基于 numpy_impl/network 单样本版）。

功能概览与 numpy_impl/test_batch.py 一致，区别在于本脚本使用单样本版
network（逐样本 / mini-batch 累加梯度）。

注意：当前 numpy_impl/network.py 存在两处已知问题，需先修复才能跑通：
  1. network.py:38  `train_step` 的 `accumulate=,` 缺参数值（语法错误）。
  2. layer.py:64   单样本时 `self.X.T @ delta` 退化为标量，dW 维度错误。
实际训练建议使用 numpy_impl/network_batch（批次版，已可用）。

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
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"图像文件魔数错误: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)

def load_idx_labels(path):
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"标签文件魔数错误: {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8)


# ------------------------- 2. 基本信息展示 -------------------------
def show_dataset_info(x, y):
    """打印数据集的形状、像素范围、各类别样本数等基本信息。"""
    print(">> 数据集基本信息：")
    print(f"  图像 shape :  {x.shape}  (N, H, W)")
    print(f"  标签 shape : {y.shape}  (N,)")
    print(f"  像素范围   : [{x.min()}, {x.max()}]")
    print(f"  类别数     : {len(np.unique(y))}")
    counts = np.bincount(y, minlength=int(y.max()) + 1)
    for c, cnt in enumerate(counts):
        print(f"    类别 {c}: {cnt} 张")


# ------------------------- 3. 可视化 -------------------------
def visualize_samples(x, y, n=10, seed=42):
    """用 matplotlib 展示前 n 张样本及其标签（无 GUI 环境下静默跳过）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无显示环境也能保存
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (跳过可视化：{e})")
        return

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(x), size=min(n, len(x)), replace=False)
    fig, axes = plt.subplots(1, len(idx), figsize=(len(idx) * 1.2, 1.4))
    if len(idx) == 1:
        axes = [axes]
    for ax, i in zip(axes, idx):
        ax.imshow(x[i], cmap="gray")
        ax.set_title(f"{y[i]}")
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join(PROJECT_ROOT, "numpy_impl_mnist_samples.png")
    fig.savefig(out, dpi=120)
    print(f"  (样本图已保存至 {out})")


# ------------------------- 4. 预处理 -------------------------
def preprocess(x, y, num_classes=10):
    """归一化像素到 [0,1]，标签转 one-hot。返回 (x_flat[N,784], y_onehot[N,10])。"""
    x_norm = x.reshape(x.shape[0], -1).astype(np.float32) / 255.0
    y_onehot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    y_onehot[np.arange(y.shape[0]), y] = 1.0
    return x_norm, y_onehot


# ------------------------- 5. 训练与测试 -------------------------
def main():
    # 1) 加载并解析数据
    print(">> 加载并解析 MNIST 数据 ...")
    x_train = load_idx_images(TRAIN_IMAGES)
    y_train = load_idx_labels(TRAIN_LABELS)
    x_test = load_idx_images(TEST_IMAGES)
    y_test = load_idx_labels(TEST_LABELS)
    show_dataset_info(x_train, y_train)

    # 2) 可视化部分样本
    visualize_samples(x_train, y_train, n=10)

    # 3) 预处理
    print(">> 数据归一化 + one-hot ...")
    x_train, y_train = preprocess(x_train, y_train)
    x_test, y_test = preprocess(x_test, y_test)

    # 4) 从训练集切出验证集，剩余用于训练
    val_size = 5000
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_tr, y_tr = x_train[val_size:], y_train[val_size:]
    print(f"train: {x_tr.shape}, val: {x_val.shape}, test: {x_test.shape}")

    # 5) 构建单样本版 MLP 并训练
    #    注：network 当前存在已知 bug（见文件头），以下为对应调用方式。
    net = network(input_size=784, hidden_dim=128,
                  num_hidden_layers=1, output_size=10)
    print(">> 开始训练（单样本版：batch_size=32, lr=0.005, momentum=0.9）...")
    net.train(x_tr, y_tr, epochs=10, lr=0.005,
              batch_size=32, momentum=0.9,
              x_val=x_val, y_val=y_val, eval_every=1)

    # 6) 测试集评估
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
