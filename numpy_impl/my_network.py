"""实验版 network（my_network）：用 hidden_shape 列表定义层结构（单样本版）。

与 network.py 的差异（实验性，可共存）：
  - 构造参数用 hidden_shape 列表（如 [128, 64, 10]）描述各隐藏层维度，
    最后一维须等于 output_size；层结构由列表链式自动搭建。
  - 其他前向/反向/训练循环逻辑与 network.py 一致（逐样本 / mini-batch 累加梯度）。

依赖 layer.py / loss.py（1D 约定）。
"""

import random

import numpy as np

from .layer import layer
from .loss import cross_entropy_grad, cross_entropy_loss


class network:
    def __init__(self, input_size, output_size, hidden_shape, print_diag=True):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_shape = list(hidden_shape)
        self.layers = []

        # hidden_shape 是每层 hidden_dim 的列表（如 [128, 64] 表示两个隐藏层）。
        # 按列表链式搭建隐藏层：input -> h0 -> h1 -> ...
        prev = input_size
        for h in self.hidden_shape:
            self.layers.append(layer(prev, h))
            prev = h
        # 末尾补一个输出层（softmax），接在最后一个隐藏层之后
        self.layers.append(layer(prev, output_size, is_output=True))

    def predict(self, X):
        out = X
        for l in self.layers:
            out = l.forward(out)
        return out

    def train_step(self, X, Y_true, accumulate=False):
        """单步前向 + 反向。accumulate=True 时梯度累加（mini-batch 用），返回损失。"""
        pred = self.predict(X)
        loss = cross_entropy_loss(pred, Y_true)
        grad = cross_entropy_grad(pred, Y_true)
        # 反向传播
        for l in self.layers:
            grad = l.backward(grad, accumulate) # accumulate: 是否要梯度累加
        return loss

    def train(self, X_date, Y_data, lr = 0.05, momentum = 0.8, 
             epochs, batch_size = 5, lr_decay = 0.98, seed = 42,
             X_val = None, Y_val = None, eval_every = 2):
        if seed is not None:
            random.seed(seed)

        history = []
        n = len(X_data)
        for epoch in range(epochs):
            idx = list(range(n))
            random.shuffle(idx)
            # 所有batch总损失
            total_loss = 0.0
            i = 0
            while i < n:
                b_idx = idx[i:i + batch_size]
                i += batch_size
                for l in self.layers:
                    l.zero_grad()
                batch_loss = 0.0
                for j in b_idx:
                    batch_loss += self.train_step(X_data[j], Y_data[j], accumulate=True)
                # 每批次训练完更新一回（按 batch 大小平均学习率）
                for l in self.layers:
                    l.update(lr / batch_size, momentum) 
                total_loss += batch_loss

            avg_loss = total_loss / n
            history.append(avg_loss)
            msg = f"epoch {epoch + 1} / {epochs}  loss = {avg_loss:.6f}"
            if X_val is not None and Y_val is not None and (epoch + 1) % eval_every == 0:
                eval_loss, acc = self.evaluate(X_val, Y_val)
                msg += f"eval_loss = {eval_loss:.6f} val_acc = {acc:.4f}"
            print(msg)
            lr *= lr_decay

        return history

    def evaluate(self, X_val, Y_val):
        total_loss = 0.0
        correct = 0
        for X, Y_true in zip(X_val, Y_val):
            pred = self.predict(X)
            total_loss += cross_entropy_loss(pred, Y_true)
            if np.argmax(pred) == np.argmax(Y_true):
                correct += 1
        n = len(X_val)
        return total_loss / n, correct / n
