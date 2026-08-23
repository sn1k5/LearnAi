"""批次版单层（layer_batch）：按 [N, features] 行样本批量前向/反向（numpy 版）。

与 layer.py 的差异（新增文件，二者可共存，不覆盖旧文件）：
  - 输入 X 形状 [N, input_size]，N 为一批的样本数（行样本，batch 导向）。
  - softmax 沿 axis=1（跨类别），适配行样本。
  - 反向梯度已按 batch 平均（mean 约定），与 PyTorch 默认一致，数值更稳。

职责边界与 layer.py 一致：
  - 只关心『一次前向 / 一次反向』的计算，以及权重/梯度/缓存的存储与初始化。
  - 参数更新（SGD + momentum）复用 optimizer.py 的 update/zero_grad。
  - 训练期诊断监控（diag）后续迭代补充（见 TODO）。
"""

import numpy as np

from .optimizer import zero_grad as _zero_grad
from .optimizer import update as _update


class layer_batch:
    def __init__(self, input_size, output_size, is_output=False):
        self.input_size = input_size
        self.output_size = output_size
        self.is_output = is_output
        # W.shape = [input_size, output_size]；He 初始化（适配 ReLU）
        std = np.sqrt(2.0 / input_size)
        self.W = np.random.randn(input_size, output_size) * std
        self.B = np.zeros(output_size)
        # 前向缓存（保留整批）
        self.X = None                             # [N, input_size]
        self.Z = None                             # [N, output_size]
        self.derivative_activation = None         # [N, output_size]
        # 梯度（mean 约定：每个 batch 反向后已除以 N）
        self.dW = np.zeros((input_size, output_size))
        self.dB = np.zeros(output_size)
        self.dX = None                            # [N, input_size]，回传给上一层
        # momentum 速度项（跨 batch/epoch 保留）
        self.vW = np.zeros((input_size, output_size))
        self.vB = np.zeros(output_size)

    # ------------------------- 激活 -------------------------
    def _activation(self, Z):
        return np.maximum(0, Z)

    # ------------------------- 前向 -------------------------
    def forward(self, X):
        """前向：X=[N, input_size] -> 激活输出 [N, output_size]（B 沿行广播）。"""
        self.X = X
        self.Z = X @ self.W + self.B
        if self.is_output:
            return self._softmax(self.Z)
        self.derivative_activation = (self.Z >= 0).astype(float)
        return self._activation(self.Z)

    # ------------------------- 反向 -------------------------
    def backward(self, delta):
        """反向：delta=[N, output_size]（输出层为 pred - y_true）。

        梯度按 batch 平均（mean 约定）：dB/dW 除以 N，回传 dX 类似层。
        因为每批反向即写入全新梯度（非累加），网络侧无需额外 zero_grad。
        """
        if not self.is_output:
            delta = delta * self.derivative_activation
        n = delta.shape[0]
        # mean 约定：除以 batch 大小，避免梯度随 batch 规模增长、稳定收敛
        self.dB = np.sum(delta, axis=0) / n
        self.dW = (self.X.T @ delta) / n
        self.dX = delta @ self.W.T
        return self.dX

    # ------------------------- 委托：参数更新 -------------------------
    def zero_grad(self):
        """清空累积梯度（保留接口，批次模式实际由 backward 覆盖写入）。"""
        _zero_grad(self)

    def update(self, lr, momentum=0.0):
        """SGD 更新（含 momentum），实现见 optimizer。"""
        _update(self, lr, momentum)

    # ------------------------- 输出层 softmax（沿类别方向 axis=1） -------------------------
    def _softmax(self, Z):
        m = np.max(Z, axis=1, keepdims=True)
        e = np.exp(Z - m)
        return e / np.sum(e, axis=1, keepdims=True)

    # ------------------------- TODO：后续迭代补充 -------------------------
    # TODO: 训练期诊断监控（死亡神经元比例 / 梯度范数 / monitor），复用 diag.py 并对 batch 维度适配。
    # TODO: 激活函数可配置（当前硬编码 ReLU / softmax 输出）。
    # TODO: L2 正则、Dropout 等正则化扩展点。