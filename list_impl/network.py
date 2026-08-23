import json
import os
import random

from .layer import layer
from .loss import cross_entropy_loss, cross_entropy_grad


class network:
    def __init__(self, input_size, hidden_dim, num_hidden_layers, output_size):
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.output_size = output_size
        self.layers = []

        prev_size = input_size
        # 隐藏层（ReLU）：num_hidden_layers 层，每层 hidden_dim 个神经元
        for _ in range(num_hidden_layers):
            self.layers.append(layer(prev_size, hidden_dim))
            prev_size = hidden_dim
        # 输出层（softmax）
        self.layers.append(layer(prev_size, output_size, is_output=True))

    def predict(self, x):
        """完整前向传播，返回输出层的 softmax 概率"""
        out = x
        for l in self.layers:
            out = l.forward(out)
        return out

    def train_step(self, x, y_true, lr=0.001, update=True, momentum=0.0):
        # 前向传播
        pred = self.predict(x)
        # 计算交叉熵损失
        loss = cross_entropy_loss(pred, y_true)
        # 反向传播：交叉熵 + softmax 合并梯度 = pred - y_true（即 z 处梯度，
        # 输出层 backward 内部不再乘激活导数），从输出层往输入层逐层回传
        grad = cross_entropy_grad(pred, y_true)
        for l in reversed(self.layers):
            grad = l.backward(grad, accumulate=not update)
        # 参数更新（逐样本时立即更新；mini-batch 时由 train 统一更新）
        if update:
            for l in self.layers:
                l.update(lr, momentum)
        return loss

    def train(self, x_data, y_data, epochs, lr=0.001,
              batch_size=1, lr_decay=1.0, seed=42, momentum=0.9,
              x_val=None, y_val=None, eval_every=1, early_stop=0,
              save_best=None):
        """训练骨架：逐样本 / mini-batch SGD，支持洗牌、衰减、定期验证、早停、模型保存。
        参数:
          batch_size : 每批样本数，1 即逐样本 SGD；>1 走 mini-batch 平均梯度。
          lr_decay   : 每轮学习率乘子（<1 逐步衰减），1.0 表示不衰减。
          seed       : 随机种子，固定可复现；None 则不固定。
          momentum   : SGD momentum 系数 β（默认 0.9），0 即普通 SGD。
          x_val/y_val: 验证集，提供后按 eval_every 周期打印验证准确率。
          eval_every : 每多少轮做一次验证（需提供验证集）。
          early_stop : 连续多少轮验证精度无提升则停止（0 表示不早停）。
          save_best  : 验证精度创新高时自动保存模型到该路径（None 表示不保存）。
        """
        print("当前参数为 lr={}, batch_size={}, lr_decay={}, seed={}, momentum={}, early_stop={} "
              .format(lr, batch_size, lr_decay, seed, momentum, early_stop))

        if seed is not None:
            random.seed(seed)

        history = []
        n = len(x_data)
        best_acc = -1.0
        best_epoch = 0
        patience = 0
        self._snapshot = None          # 最佳权重快照（深拷贝），用于早停回滚
        for epoch in range(epochs):
            idx = list(range(n))
            random.shuffle(idx)                      # 每轮洗牌：打破固定顺序相关性

            total_loss = 0.0
            i = 0
            while i < n:
                # 取本 batch 的样本下标
                b_idx = idx[i:i + batch_size]
                i += batch_size
                # mini-batch：清零旧梯度，累加该批各样本梯度，再按批量平均更新一次
                for l in self.layers:
                    l.zero_grad()
                batch_loss = 0.0
                for j in b_idx:
                    batch_loss += self.train_step(x_data[j], y_data[j], lr,
                                                  update=False, momentum=momentum)
                # 用平均梯度统一更新一次（含 momentum）
                for l in self.layers:
                    l.update(lr / len(b_idx), momentum)
                total_loss += batch_loss

            avg_loss = total_loss / n
            history.append(avg_loss)
            msg = f"epoch {epoch + 1}/{epochs}  loss = {avg_loss:.6f}"
            # 验证 + 早停 + 最佳保存判断
            if x_val is not None and y_val is not None and (epoch + 1) % eval_every == 0:
                _, acc = self.evaluate(x_val, y_val)
                msg += f"  val_acc = {acc:.4f}"
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch + 1
                    patience = 0
                    self._snapshot = self._deepcopy_weights()   # 记下最好权重
                    if save_best:
                        self.save(save_best)
                        msg += f"  [保存最佳 {save_best}]"
                else:
                    patience += 1
            print(msg)

            # 每轮打印各层监控（死亡比例 / 权重范数 / 梯度范数 / 激活均值）
            for li, l in enumerate(self.layers):
                m = l.monitor()
                dead = f" dead={m['dead']:.2%}" if "dead" in m else ""
                print(f"    [L{li}] w_norm={m['w_norm']:.3f} "
                      f"g_norm={m['g_norm']:.4f} act_mean={m['act_mean']:.4f}{dead}")

            lr *= lr_decay                       # 学习率衰减

            # 早停：连续 early_stop 轮无提升则回滚最佳权重并停止
            if early_stop > 0 and patience >= early_stop:
                print(f"  >> 早停：连续 {patience} 轮验证精度无提升，"
                      f"回滚至第 {best_epoch} 轮最佳权重 (val_acc={best_acc:.4f})")
                if self._snapshot is not None:
                    self._restore_weights(self._snapshot)
                break
        else:
            # 正常跑完：若开早停且曾记录过更好权重，回滚到最佳（而非最后一轮）
            if early_stop > 0 and self._snapshot is not None:
                print(f"  >> 训练结束，回滚至第 {best_epoch} 轮最佳权重 "
                      f"(val_acc={best_acc:.4f})")
                self._restore_weights(self._snapshot)
        return history

    # ---------- 模型保存 / 加载 / 权重快照 ----------
    def _deepcopy_weights(self):
        """深拷贝所有层的 w/b/v_w/v_b，供早停回滚与保存前快照使用。"""
        snap = []
        for l in self.layers:
            snap.append({
                "w": [row[:] for row in l.w],
                "b": l.b[:],
                "v_w": [row[:] for row in l.v_w],
                "v_b": l.v_b[:],
            })
        return snap

    def _restore_weights(self, snap):
        for l, s in zip(self.layers, snap):
            l.w = [row[:] for row in s["w"]]
            l.b = s["b"][:]
            l.v_w = [row[:] for row in s["v_w"]]
            l.v_b = s["v_b"][:]

    def save(self, path):
        """将模型结构 + 权重保存到 JSON 文件，可后续 load 继续训练。"""
        state = {
            "input_size": self.input_size,
            "hidden_dim": self.hidden_dim,
            "num_hidden_layers": self.num_hidden_layers,
            "output_size": self.output_size,
            "layers": [{
                "is_output": l.is_output,
                "w": l.w, "b": l.b, "v_w": l.v_w, "v_b": l.v_b,
            } for l in self.layers],
        }
        with open(path, "w") as f:
            json.dump(state, f)
        print(f"  >> 模型已保存: {path}")

    @classmethod
    def load(cls, path):
        """从 JSON 文件加载模型（结构 + 权重 + 动量速度项），返回 network 实例。"""
        with open(path, "r") as f:
            state = json.load(f)
        net = cls(state["input_size"], state["hidden_dim"],
                  state["num_hidden_layers"], state["output_size"])
        for l, s in zip(net.layers, state["layers"]):
            l.w = s["w"]
            l.b = s["b"]
            l.v_w = s["v_w"]
            l.v_b = s["v_b"]
        print(f"  >> 模型已加载: {path}")
        return net

    def monitor_layers(self):
        """返回各层监控指标列表（dict），每个元素对应 self.layers 的一层。"""
        return [l.monitor() for l in self.layers]

    def batch_dead_ratios(self, x_list):
        """返回各层基于整批样本统计的整体/永久死亡比例列表（输出层为 None）。"""
        return [l.dead_ratio_over_batch(x_list) for l in self.layers]

    def evaluate(self, x_data, y_data):
        """返回 (平均损失, 准确率)"""
        total_loss = 0.0
        correct = 0
        for x, y_true in zip(x_data, y_data):
            pred = self.predict(x)
            total_loss += cross_entropy_loss(pred, y_true)
            if pred.index(max(pred)) == y_true.index(max(y_true)):
                correct += 1
        n = len(x_data)
        return total_loss / n, correct / n
