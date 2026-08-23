"""list 版 MLP 实现包。

用纯 Python list 手写多层感知机（不依赖 numpy）：
  - layer.py      : 单层（含 ReLU / softmax、前向、反向、参数更新）
  - loss.py       : 交叉熵损失与梯度
  - network.py    : 功能完整版网络（早停 / 验证 / 保存加载）
  - my_network.py : 精简版网络（仅 SGD 训练骨架）

未来可新增 numpy_impl/ 作为 numpy 版实现，结构与此包对齐。
"""

from .layer import layer
from .loss import cross_entropy_loss, cross_entropy_grad
from .layer_update import zero_grad, update
from .layer_diag import (
    monitor,
    dead_neuron_ratio,
    dead_ratio_over_batch,
    weight_norm,
    grad_norm,
    activation_mean,
)
from .network import network
from .my_network import network as my_network

__all__ = [
    "layer",
    "cross_entropy_loss",
    "cross_entropy_grad",
    "zero_grad",
    "update",
    "monitor",
    "dead_neuron_ratio",
    "dead_ratio_over_batch",
    "weight_norm",
    "grad_norm",
    "activation_mean",
    "network",
    "my_network",
]
