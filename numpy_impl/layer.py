"""单层（layer）：前向传播、反向传播、参数初始化（numpy 版）。

职责边界与 list_impl.layer 一致：
  - 只关心『一次前向 / 一次反向』的计算，以及权重/梯度/缓存的存储与初始化。
  - 参数更新（SGD + momentum）见 optimizer.py
  - 训练期诊断监控（死亡比例/范数/monitor）见 diag.py
"""

import numpy as np

from .optimizer import zero_grad as _zero_grad
from .optimizer import update as _update
from .diag import (
    monitor as _monitor,
    dead_neuron_ratio as _dead_neuron_ratio,
    dead_ratio_over_batch as _dead_ratio_over_batch,
)


class layer:
    def __init__(self, input_size, output_size, is_output=False):
        self.input_size = input_size
        self.output_size = output_size
        self.is_output = is_output
        # W.shape = [input_size, output_size]
        # He 初始化（适配 ReLU）
        std = np.sqrt(2.0 / input_size)
        self.W = np.random.randn(input_size, output_size) * std
        self.B = np.zeros(output_size)
        self.X = np.zeros(input_size)
        self.Z = np.zeros(output_size)
        self.derivative_activation = np.zeros(output_size)
        self.dW = np.zeros((input_size, output_size))
        self.dB = np.zeros(output_size)
        self.dX = np.zeros(input_size)
        # momentum 速度项（与 W/B 同形），跨 batch/epoch 保留
        self.vW = np.zeros((input_size, output_size))
        self.vB = np.zeros(output_size)

    # ------------------------- 激活 -------------------------
    def _activation(self, Z):
        return np.maximum(0, Z)

    def _save_derivative_activation(self, Z):
        self.derivative_activation = (Z >= 0).astype(float)

    # ------------------------- 前向 -------------------------
    def forward(self, X):
        # 输入统一为 1D [input_size]，Z 与输出均保持 1D [output_size]
        self.X = np.asarray(X).reshape(self.input_size)
        self.Z = self.X @ self.W + self.B
        if self.is_output:
            return self._softmax(self.Z)
        self._save_derivative_activation(self.Z)
        return self._activation(self.Z)

    # ------------------------- 反向 -------------------------
    def backward(self, delta, accumulate=False):
        # 输出层：delta 已是 softmax 输入处的梯度，不再乘激活导数
        if not self.is_output:
            delta = delta * self.derivative_activation
        # 更新 B 梯度（delta 为 1D [output]）
        self.dB = self.dB + delta if accumulate else delta
        # 更新 W 梯度：X[input] 与 delta[output] 的外积 -> [input, output]
        dW = np.outer(self.X, delta)
        self.dW = self.dW + dW if accumulate else dW
        # 回传梯度给前一层：W[input,output] @ delta[output] = [input]
        self.dX = self.W @ delta
        return self.dX

    # ------------------------- 委托：参数更新 -------------------------
    def zero_grad(self):
        """清空累积梯度，每个 mini-batch 开始前调用（实现见 optimizer）。"""
        _zero_grad(self)

    def update(self, lr, momentum=0.0):
        """SGD 更新（含 momentum），实现见 optimizer。"""
        _update(self, lr, momentum)

    # ------------------------- 委托：诊断监控 -------------------------
    def dead_neuron_ratio(self):
        """瞬时死亡神经元比例（实现见 diag）。"""
        return _dead_neuron_ratio(self)

    def dead_ratio_over_batch(self, X_list):
        """整批永久死亡比例（实现见 diag）。"""
        return _dead_ratio_over_batch(self, X_list)

    def monitor(self):
        """汇总本层监控指标（实现见 diag）。"""
        return _monitor(self)

    # ------------------------- 输出层 softmax -------------------------
    def _softmax(self, Z):
        m = np.max(Z, axis=0, keepdims=True)
        e = np.exp(Z - m)
        return e / np.sum(e, axis=0, keepdims=True)
