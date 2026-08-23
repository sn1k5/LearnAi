"""批次版神经网络（network_batch）：组装 batch layer + batch loss + 完整训练闭环。

按 [N, features] 整批前向/反向，mean 平均梯度；与前单样本版（network.py）新增文件，可共存。
功能：
  - predict  : 整批预测（softmax 概率）。
  - train_step : 单批前向 + 反向 + 更新，返回该批平均损失。

  - train    : epoch + shuffle + mini-batch + lr 衰减 + 定期验证。
  - evaluate : 整批评估（平均损失 + 准确率）。

TODO（后续迭代）：
  - 模型保存 / 加载（JSON）。
  - 训练期诊断监控（diag）。
  - 完整参数校验 / 异常处理 / 末尾不完整 batch 的语义与告警。
  - 学习率调度策略扩展点。
"""

import random

import numpy as np

from .layer_batch import layer_batch
from .loss_batch import cross_entropy_loss, cross_entropy_grad


def _to_ndarray(data, dtype=None):
    """把 list/array 统一为 ndarray；data 为 list 里的行样本时 np.array 自动堆叠成 [N, D]。"""
    return np.asarray(data, dtype=dtype)


class network_batch:
    def __init__(self, input_size, hidden_dim, num_hidden_layers, output_size):
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.output_size = output_size
        self.layers = []

        prev = input_size
        # 隐藏层（ReLU，批次版）
        for _ in range(num_hidden_layers):
            self.layers.append(layer_batch(prev, hidden_dim))
            prev = hidden_dim
        # 输出层（softmax，批次版）
        self.layers.append(layer_batch(prev, output_size, is_output=True))

    def predict(self, X):
        """完整前向传播，返回输出层 softmax 概率 [N, output_size]。"""
        out = _to_ndarray(X, dtype=float)
        for l in self.layers:
            out = l.forward(out)
        return out

    def train_step(self, X, Y_true, lr=0.001, momentum=0.0):
        """单批前向 + 反向 + 更新，返回该批平均损失（标量）。"""
        pred = self.predict(X)
        loss = cross_entropy_loss(pred, _to_ndarray(Y_true))
        grad = cross_entropy_grad(pred, _to_ndarray(Y_true))
        for l in reversed(self.layers):
            grad = l.backward(grad)
        for l in self.layers:
            l.update(lr, momentum)
        return loss

    def train(self, x_data, y_data, epochs, lr=0.001,
              batch_size=32, lr_decay=1.0, seed=42, momentum=0.9,
              x_val=None, y_val=None, eval_every=1):
        """批次训练闭环：整批前向/反向/更新，mean 平均梯度，支持洗牌、衰减、定期验证。"""
        if seed is not None:
            random.seed(seed)

        x_data = _to_ndarray(x_data, dtype=float)
        y_data = _to_ndarray(y_data)
        history = []
        n = len(x_data)
        for epoch in range(epochs):
            idx = list(range(n))
            random.shuffle(idx)

            epoch_loss = 0.0
            # 末尾不足 batch_size 的余批也照常训练（batch 很小，mean 约定下仍稳定）
            for i in range(0, n, batch_size):
                b_idx = idx[i:i + batch_size]
                Xb = x_data[b_idx]
                Yb = y_data[b_idx]
                # 每批损失即平均损失，按该批样本数加权求和以获得 epoch 平均
                epoch_loss += self.train_step(Xb, Yb, lr, momentum) * len(b_idx)

            avg_loss = epoch_loss / n
            history.append(avg_loss)
            msg = f"epoch {epoch + 1}/{epochs}  loss = {avg_loss:.6f}"
            if x_val is not None and y_val is not None and (epoch + 1) % eval_every == 0:
                _, acc = self.evaluate(x_val, y_val)
                msg += f"  val_acc = {acc:.4f}"
            print(msg)

            lr *= lr_decay
        return history

    def evaluate(self, x_data, y_data, batch_size=256):
        """返回 (平均损失, 准确率)。分批前向以控制峰值内存，mean 加权聚合。"""
        X = _to_ndarray(x_data, dtype=float)
        Y = _to_ndarray(y_data)
        n = len(X)
        total_loss = 0.0
        correct = 0
        for i in range(0, n, batch_size):
            Xb = X[i:i + batch_size]
            Yb = Y[i:i + batch_size]
            preds = self.predict(Xb)
            total_loss += cross_entropy_loss(preds, Yb) * len(Xb)
            correct += int(np.sum(np.argmax(preds, axis=1) == np.argmax(Yb, axis=1)))
        return total_loss / n, correct / n