"""MNIST 手写数字识别 —— 测试脚本（基于 list_impl 纯 list 实现）。

功能概览：
  1. 加载并解析本地 MNIST 训练集（由 download_mnist.py 下载）。
  2. 展示数据集基本信息：样本数、图片尺寸、类别数等。
  3. 随机抽取若干样本并可视化显示。
  4. 数据预处理：将像素 [0,255] 归一化到 [0,1]，标签转 one-hot。
  5. 使用 list_impl 中手写的简单神经网络（network）进行训练与测试，输出准确率。

运行方式：
  python download_mnist.py              # 首次需先下载数据
  python list_impl/test_mnist.py        # 注意：经 list_impl 包导入，需从项目根运行
"""

import os
import random
import struct
import sys

import matplotlib.pyplot as plt
import numpy as np

# 将项目根加入 sys.path，使 `from list_impl.network import ...` 可解析
PROJECT_ROOT = "c:/Users/Administrator/Desktop/Code/Proj/Ai/myMLP"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from list_impl.network import network  # 复用 list 版手写 MLP 框架

# ------------------------- 路径配置（写死绝对路径） -------------------------
DATA_DIR = "c:/Users/Administrator/Desktop/Code/Proj/Ai/myMLP/data"
TRAIN_IMAGES = os.path.join(DATA_DIR, "train-images-idx3-ubyte")
TRAIN_LABELS = os.path.join(DATA_DIR, "train-labels-idx1-ubyte")
TEST_IMAGES = os.path.join(DATA_DIR, "t10k-images-idx3-ubyte")
TEST_LABELS = os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte")


# ------------------------- 1. 解析 IDX 格式 -------------------------
def load_idx_images(path):
    """解析 IDX3 图像文件，返回形状 (N, H, W) 的 uint8 数组。"""
    with open(path, "rb") as f:
        # 魔数(4字节) + 图片数(4) + 行数(4) + 列数(4)
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


# ------------------------- 2. 基本信息展示 -------------------------
def show_dataset_info(x, y):
    """打印数据集基本信息。"""
    print("=" * 50)
    print("MNIST 数据集基本信息")
    print("=" * 50)
    print(f"样本数量 : {x.shape[0]}")
    print(f"图片尺寸 : {x.shape[1]} x {x.shape[2]} 像素 (灰度)")
    print(f"像素范围 : [{x.min()}, {x.max()}]")
    print(f"类别数量 : {len(np.unique(y))} (数字 0~9)")
    print(f"类别分布 :", dict(zip(*np.unique(y, return_counts=True))))
    print("=" * 50)


# ------------------------- 3. 可视化 -------------------------
def visualize_samples(x, y, n=10, seed=42):
    """随机抽取 n 个样本，以网格形式可视化显示。"""
    rng = random.Random(seed)
    indices = rng.sample(range(len(x)), n)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.2, 1.4))
    for ax, idx in zip(axes, indices):
        ax.imshow(x[idx], cmap="gray")
        ax.set_title(f"label={y[idx]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("MNIST 随机样本可视化", fontsize=12)
    fig.tight_layout()
    plt.show()


# ------------------------- 4. 预处理 -------------------------
def preprocess(x, y, num_classes=10):
    """归一化像素到 [0,1]，并将标签转为 one-hot 向量。

    返回: (x_flat, y_onehot)，x_flat 形状 (N, 784)，y_onehot 形状 (N, 10)。
    """
    # 归一化：除以 255 将 [0,255] 映射到 [0,1]，加速收敛、稳定梯度
    x_norm = x.reshape(x.shape[0], -1).astype(np.float32) / 255.0
    # one-hot 编码
    y_onehot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    y_onehot[np.arange(y.shape[0]), y] = 1.0
    return x_norm, y_onehot


def to_python_lists(x, y):
    """将 numpy 数组转为 Python list，适配手写 MLP 的输入格式。"""
    return x.tolist(), y.tolist()


# ------------------------- 5. 训练与测试 -------------------------
def main():
    # 1) 加载并解析数据
    print(">> 加载并解析 MNIST 数据 ...")
    x_train = load_idx_images(TRAIN_IMAGES)
    y_train = load_idx_labels(TRAIN_LABELS)
    x_test = load_idx_images(TEST_IMAGES)
    y_test = load_idx_labels(TEST_LABELS)

    # 2) 基本信息
    show_dataset_info(x_train, y_train)

    # 3) 随机可视化
    print(">> 随机抽取样本可视化 ...")
    visualize_samples(x_train, y_train, n=10)

    # 为控制训练时长，这里仅用部分训练/测试样本（可改 full 为 True 用全量）
    USE_SUBSET = True
    subset_train, subset_test = 1000, 1000
    if USE_SUBSET:
        x_train, y_train = x_train[:subset_train], y_train[:subset_train]
        x_test, y_test = x_test[:subset_test], y_test[:subset_test]
        print(f">> 使用子集训练={len(x_train)} 测试={len(x_test)}（如需全量请改 USE_SUBSET=False）")

    # 4) 预处理
    x_train_n, y_train_o = preprocess(x_train, y_train)
    x_test_n, y_test_o = preprocess(x_test, y_test)
    x_tr, y_tr = to_python_lists(x_train_n, y_train_o)
    x_te, y_te = to_python_lists(x_test_n, y_test_o)

    # 5) 构建并训练手写 MLP
    print(">> 构建并训练简单神经网络 (手写 MLP) ...")
    model = network(
        input_size=28 * 28,     # 输入：784 个像素
        hidden_dim=128,         # 隐藏层维度
        num_hidden_layers=2,    # 隐藏层数量
        output_size=10,         # 输出：10 个类别
    )

    # --- 初始化诊断：训练前先看第一隐藏层的死亡神经元比例 ---
    model.predict(x_tr[0])                       # 触发一次前向，填充各层 self.z
    for idx, l in enumerate(model.layers):
        ratio = l.dead_neuron_ratio()
        if ratio is not None:
            print(f"[诊断] 第{idx}层 ReLU 死亡神经元比例: {ratio:.2%}")

    model.train(
        x_tr, y_tr,
        epochs=10,
        lr=0.1,
        batch_size=64,
        lr_decay=0.999,
        seed=42,
        x_val=x_te,
        y_val=y_te,
        eval_every=2,
    )

    # --- 训练后诊断：死亡比例应显著下降，若仍接近 1.0 说明有层彻底死了 ---
    model.predict(x_tr[0])
    for idx, l in enumerate(model.layers):
        ratio = l.dead_neuron_ratio()
        if ratio is not None:
            print(f"[诊断] 训练后第{idx}层死亡神经元比例: {ratio:.2%}")

    # 输出测试准确率
    _, acc = model.evaluate(x_te, y_te)
    print("\n" + "=" * 50)
    print(f"测试集准确率 (Accuracy): {acc * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
