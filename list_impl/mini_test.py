"""最简 MNIST 测试：加载本地 data/ 中的 MNIST，训练并测试 list_impl 手写 MLP。

运行前请确保已下载数据：python download_mnist.py
直接运行即可：python list_impl/mini_test.py
"""

import os
import struct
import sys

import numpy as np

# 将项目根加入 sys.path，使 `from list_impl.network import ...` 可解析
PROJECT_ROOT = "c:/Users/Administrator/Desktop/Code/Proj/Ai/myMLP"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from list_impl.network import network

# ------------------------- 路径配置（写死绝对路径） -------------------------
DATA_DIR = "c:/Users/Administrator/Desktop/Code/Proj/Ai/myMLP/data"
TRAIN_IMAGES = os.path.join(DATA_DIR, "train-images-idx3-ubyte")
TRAIN_LABELS = os.path.join(DATA_DIR, "train-labels-idx1-ubyte")
TEST_IMAGES = os.path.join(DATA_DIR, "t10k-images-idx3-ubyte")
TEST_LABELS = os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte")

# 模型权重文件随 list_impl 包一起存放
MODEL_PATH = "c:/Users/Administrator/Desktop/Code/Proj/Ai/myMLP/list_impl/mlp_model.json"


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


def prepare(x, y, num_classes=10):
    x_norm = x.reshape(x.shape[0], -1).astype(np.float32) / 255.0
    y_onehot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    y_onehot[np.arange(y.shape[0]), y] = 1.0
    return x_norm.tolist(), y_onehot.tolist()


def main():
    print(">> 加载 MNIST 数据 ...")
    x_train = load_idx_images(TRAIN_IMAGES)
    y_train = load_idx_labels(TRAIN_LABELS)
    x_test = load_idx_images(TEST_IMAGES)
    y_test = load_idx_labels(TEST_LABELS)

    # 为控制训练时长，这里仅用部分训练/测试样本（可改 full 为 True 用全量）
    USE_SUBSET = True
    subset_train, subset_test = 1000, 800
    if USE_SUBSET:
        x_train, y_train = x_train[:subset_train], y_train[:subset_train]
        x_test, y_test = x_test[:subset_test], y_test[:subset_test]

    x_tr, y_tr = prepare(x_train, y_train)
    x_te, y_te = prepare(x_test, y_test)
    print(f"   训练样本 {len(x_tr)}，测试样本 {len(x_te)}")

    # 若已存在保存的模型则加载（接着训），否则新建
    if os.path.exists(MODEL_PATH):
        print(f">> 检测到已保存模型 {MODEL_PATH}，加载后继续训练 ...")
        model = network.load(MODEL_PATH)
    else:
        print(">> 新建模型 ...")
        model = network(input_size=28 * 28, hidden_dim=128,
                        num_hidden_layers=2, output_size=10)

    print(">> 训练 ...")
    model.train(x_tr, y_tr, epochs=24, lr=0.001, batch_size=64,
                lr_decay=0.9, seed=42, momentum=0.8,
                x_val=x_te, y_val=y_te, eval_every=2,
                early_stop=3,            # 连续 2 轮验证无提升则停（回滚最佳）
                save_best=MODEL_PATH)    # 验证创新高即保存，方便下次接着训

    _, acc = model.evaluate(x_te, y_te)

    # 整体/永久死亡比例：用整批训练样本统计（而非单样本瞬时值）
    print("\n>> 整体死亡比例（基于整批训练样本）:")
    for li, r in enumerate(model.batch_dead_ratios(x_tr)):
        if r is None:
            print(f"   [L{li}] 输出层，跳过")
        else:
            print(f"   [L{li}] perm_dead = {r:.2%}")

    print("\n" + "=" * 40)
    print(f"测试准确率: {acc * 100:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()
