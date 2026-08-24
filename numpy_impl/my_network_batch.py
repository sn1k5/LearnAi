import random

import numpy as np

from .my_layer_batch import my_layer_batch as layer_batch
from .loss_batch import cross_entropy_loss, cross_entropy_grad
from .saveable import _SaveableMixin


class my_network_batch(_SaveableMixin):
    def __init__(self, input_size, hidden_shape_list, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_shape_list = hidden_shape_list
        self.layers = []

        prev = input_size
        for s in hidden_shape_list:
            self.layers.append(layer_batch(prev, s, is_output=False))
            prev = s
        self.layers.append(layer_batch(prev, output_size, is_output=True))

    def _build_from_shapes(self, layer_shapes):
        """依据 (input_size, output_size, is_output) 元组列表重建层（my_layer_batch 约定）。"""
        self.layers = [layer_batch(*shape) for shape in layer_shapes]

    def predict(self, X):
        out = X
        for l in self.layers:
            out = l.forward(out)
        return out

    def train_step(self, X, Y_true, lr=0.001, momentum=0.0):
        """单批前向 + 反向 + 更新，返回该批平均损失（标量）。"""
        pred = self.predict(X)
        loss = cross_entropy_loss(pred, Y_true)
        grad = cross_entropy_grad(pred, Y_true)
        for l in reversed(self.layers):
            grad = l.backward(grad)
        for l in self.layers:
            l.update(lr, momentum)
        return loss


    def train(self, x_data, y_data, epochs, lr=0.001,
              batch_size=32, lr_decay=1.0, seed=42, momentum=0.9,
              x_val=None, y_val=None, eval_every=1,
              monitor="val_acc", save_best=None, early_stop=None):
        if seed is not None:
            random.seed(seed)
 
        history = []
        state = self._begin_monitor(monitor)
        n = len(x_data)
        stopped = False
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

    def evaluate(self, x_data, y_data, batch_size=256):
        """返回 (平均损失, 准确率)。分批前向以控制峰值内存，mean 加权聚合。"""
        n = len(x_data)
        total_loss = 0.0
        correct = 0
        for i in range(0, n, batch_size):
            Xb = x_data[i:i + batch_size]
            Yb = y_data[i:i + batch_size]
            preds = self.predict(Xb)
            total_loss += cross_entropy_loss(preds, Yb) * len(Xb)
            correct += int(np.sum(np.argmax(preds, axis=1) == np.argmax(Yb, axis=1)))
        return total_loss / n, correct / n