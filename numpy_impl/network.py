"""神经网络（network）：组装多层 + 前向预测 + 训练循环（numpy 基础版）。

仅实现基础功能：predict / train_step / train / evaluate，不含模型保存/加载。
"""

import random

import numpy as np

from .layer import layer
from .loss import cross_entropy_loss, cross_entropy_grad
from .saveable import _SaveableMixin


class network(_SaveableMixin):
    def __init__(self, input_size, hidden_dim, num_hidden_layers, output_size, print_diag = True):
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.output_size = output_size
        self.layers = []
        self.print_diag = print_diag

        prev_size = input_size
        # 隐藏层（ReLU）
        for _ in range(num_hidden_layers):
            self.layers.append(layer(prev_size, hidden_dim))
            prev_size = hidden_dim
        # 输出层（softmax）
        self.layers.append(layer(prev_size, output_size, is_output=True))

    def _build_from_shapes(self, layer_shapes):
        """依据 (input_size, output_size, is_output) 元组列表重建层（标准 hidden_dim 约定）。"""
        self.layers = [layer(*shape) for shape in layer_shapes]

    def predict(self, x):
        """完整前向传播，返回输出层的 softmax 概率。"""
        out = x
        for l in self.layers:
            out = l.forward(out)
        return out

    def train_step(self, x, y_true, lr=0.001, accumulate=False, update=True, momentum=0.0):
        """单次前向 + 反向 + （可选）参数更新，返回损失。"""
        pred = self.predict(x)
        loss = cross_entropy_loss(pred, y_true)
        grad = cross_entropy_grad(pred, y_true)
        for l in reversed(self.layers):
            grad = l.backward(grad, accumulate=not update)
        if update:
            for l in self.layers:
                l.update(lr, momentum)
        return loss

    def train(self, x_data, y_data, epochs, lr=0.001,
              batch_size=1, lr_decay=1.0, seed=42, momentum=0.9,
              x_val=None, y_val=None, eval_every=1,
              monitor="val_acc", save_best=None, early_stop=None):
        """基础训练循环：逐样本 / mini-batch SGD，支持洗牌、衰减、定期验证。

        参数:
          batch_size : 每批样本数，1 即逐样本 SGD；>1 走 mini-batch 平均梯度。
          lr_decay   : 每轮学习率乘子（<1 逐步衰减），1.0 表示不衰减。
          seed       : 随机种子，固定可复现；None 则不固定。
          momentum   : SGD momentum 系数 β（默认 0.9），0 即普通 SGD。
          x_val/y_val: 验证集，提供后按 eval_every 周期打印验证准确率。
          eval_every : 每多少轮做一次验证（需提供验证集）。
          monitor    : 早停与 best 保存所依据的指标，'val_acc'(默认) 或 'val_loss'。
          save_best  : 路径字符串；指标刷新时自动保存当前模型为该路径（best 模型）。
          early_stop : 整数 patience；验证指标连续 patience 轮无改善则提前停止。
        """
        if seed is not None:
            random.seed(seed)

        history = []
        state = self._begin_monitor(monitor)
        n = len(x_data)
        stopped = False
        for epoch in range(epochs):
            idx = list(range(n))
            random.shuffle(idx)

            total_loss = 0.0
            i = 0
            while i < n:
                b_idx = idx[i:i + batch_size]
                i += batch_size
                # mini-batch：清零旧梯度，累加该批各样本梯度，再按批量平均更新一次
                for l in self.layers:
                    l.zero_grad()
                batch_loss = 0.0
                for j in b_idx:
                    batch_loss += self.train_step(x_data[j], y_data[j], lr, accumulate=True,
                                                  update=False, momentum=momentum)
                for l in self.layers:
                    l.update(lr / len(b_idx), momentum)
                total_loss += batch_loss

            avg_loss = total_loss / n
            history.append(avg_loss)
            msg = f"epoch {epoch + 1}/{epochs}  loss = {avg_loss:.6f}"
            if x_val is not None and y_val is not None and (epoch + 1) % eval_every == 0:
                val_loss, acc = self.evaluate(x_val, y_val)
                msg += f"  val_acc = {acc:.4f}"
                improved = self._maybe_checkpoint(
                    state, epoch, val_loss, acc, save_best, save_best
                )
                if improved:
                    msg += "  [saved best]"
                msg += self._monitor_summary(state)
                if self._should_early_stop(state, epoch, early_stop):
                    msg += f"  early_stop: best {state['monitor']} 连续 {state['no_improve']} 轮无改善"
                    print(msg)
                    stopped = True
                    break
            print(msg)

            lr *= lr_decay
        if stopped:
            print(f">>> 早停触发，最优模型来自 epoch {state['best_epoch'] + 1}"
                  + (f"，已存于 {state['best_path']}" if state['best_path'] else ""))
        return history

    def evaluate(self, x_data, y_data):
        """返回 (平均损失, 准确率)。pred / y_true 均按 1D [output] 处理。"""
        total_loss = 0.0
        correct = 0
        for x, y_true in zip(x_data, y_data):
            pred = self.predict(x)
            total_loss += cross_entropy_loss(pred, y_true)
            if np.argmax(pred) == np.argmax(y_true):
                correct += 1
        n = len(x_data)
        return total_loss / n, correct / n
