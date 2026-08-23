"""参数更新职责（numpy 版）：梯度清零与 SGD(+momentum) 权重更新。

本模块只负责『拿累积好的梯度去更新参数』。函数接收 layer 实例，
直接修改其 W/B/vW/vB。
"""

import numpy as np


def zero_grad(layer):
    """清空累积梯度，每个 mini-batch 开始前调用。"""
    layer.dW = np.zeros_like(layer.W)
    layer.dB = np.zeros_like(layer.B)


def update(layer, lr, momentum=0.0):
    """SGD 更新；momentum>0 时按 v = β·v - η·∇ 累加速度后更新权重。

    公式: v_{t+1} = β·v_t - η·∇W ;  W_{t+1} = W + v_{t+1}
    """
    layer.vW = momentum * layer.vW - lr * layer.dW
    layer.W += layer.vW
    layer.vB = momentum * layer.vB - lr * layer.dB
    layer.B += layer.vB
