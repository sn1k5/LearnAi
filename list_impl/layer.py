"""单层（layer）：前向传播、反向传播、参数初始化。

职责边界：
  - 本文件只关心『一次前向 / 一次反向』的计算，以及权重/梯度/缓存的存储与初始化。
  - 参数更新（SGD + momentum）见 layer_update.py
  - 训练期诊断监控（死亡比例/范数/monitor）见 layer_diag.py
  - layer 仍对外保留 zero_grad()/update()/monitor()/dead_*() 方法，
    内部委托到上面两个模块，因此对 network 层调用方透明、无需改动。
"""

import math
import random

from .layer_update import zero_grad as _zero_grad
from .layer_update import update as _update
from .layer_diag import (
    monitor as _monitor,
    dead_neuron_ratio as _dead_neuron_ratio,
    dead_ratio_over_batch as _dead_ratio_over_batch,
)


class layer:
    def __init__(self, input_size, output_size, is_output=False):
        self.input_size = input_size
        self.output_size = output_size
        self.is_output = is_output
        # w.shape = [input_size, output_size]
        # He 初始化（适配 ReLU）：均值 0、方差 2/input_size，
        std = math.sqrt(2.0 / input_size)
        self.w = [[random.gauss(0.0, std) for _ in range(output_size)]
                      for _ in range(input_size)]
        self.b = [0.1 for _ in range(output_size)]
        self.z = []
        self.derivative_activation = []  # 由 save_derivative_activation 在 forward 时整体重建
        self.grad_w = [[0.0 for _ in range(output_size)] for _ in range(input_size)]
        self.grad_b = [0.0 for _ in range(output_size)]
        self.grad_x = []
        self.x = []
        # momentum 速度项（与 w/b 同形），跨 batch/epoch 保留
        self.v_w = [[0.0 for _ in range(output_size)] for _ in range(input_size)]
        self.v_b = [0.0 for _ in range(output_size)]

    # ------------------------- 激活 -------------------------
    def activation(self, z):
        return [max(0, v) for v in z]

    def save_derivative_activation(self, z):
        self.derivative_activation = [1.0 if v >= 0 else 0.0 for v in z]

    # ------------------------- 前向 -------------------------
    def forward(self, x):
        self.x = x
        self.z = []
        for i in range(self.output_size):
            z_row = 0.0
            for j in range(self.input_size):
                z_row += self.w[j][i] * x[j]
            self.z.append(z_row + self.b[i])       # 顺手把 z 拍平成一维

        if self.is_output:                          # 输出层：softmax（数值稳定版）
            m = max(self.z)
            exps = [math.exp(v - m) for v in self.z]
            s = sum(exps)
            return [e / s for e in exps]
        self.save_derivative_activation(self.z)     # 保存 ReLU 导数，供 backward 使用
        return self.activation(self.z)              # 隐藏层：ReLU

    # ------------------------- 反向 -------------------------
    def backward(self, loss, accumulate=False):
        delta = []
        for i in range(len(loss)):
            # 输出层：loss 已经是 softmax 输入处的梯度，不再乘激活导数
            act = 1.0 if self.is_output else self.derivative_activation[i]
            delta.append(act * loss[i])

        for i in range(self.output_size):
            for j in range(self.input_size):
                g = delta[i] * self.x[j]
                # 非累加模式（逐样本）直接覆盖；累加模式（mini-batch）叠加
                self.grad_w[j][i] = self.grad_w[j][i] + g if accumulate else g
            gb = delta[i]
            self.grad_b[i] = self.grad_b[i] + gb if accumulate else gb

        grad_x = [0.0 for _ in range(self.input_size)]
        for j in range(self.input_size):
            for i in range(self.output_size):
                grad_x[j] += self.w[j][i] * delta[i]
        return grad_x

    # ------------------------- 委托：参数更新 -------------------------
    def zero_grad(self):
        """清空累积梯度，每个 mini-batch 开始前调用（实现见 layer_update）。"""
        _zero_grad(self)

    def update(self, lr, momentum=0.0):
        """SGD 更新（含 momentum），实现见 layer_update。"""
        _update(self, lr, momentum)

    # ------------------------- 委托：诊断监控 -------------------------
    def dead_neuron_ratio(self):
        """瞬时死亡神经元比例（实现见 layer_diag）。"""
        return _dead_neuron_ratio(self)

    def dead_ratio_over_batch(self, x_list):
        """整批永久死亡比例（实现见 layer_diag）。"""
        return _dead_ratio_over_batch(self, x_list)

    def monitor(self):
        """汇总本层监控指标（实现见 layer_diag）。"""
        return _monitor(self)