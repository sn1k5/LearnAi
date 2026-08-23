"""损失函数职责（numpy 版）：交叉熵损失与其对输出层 z 的梯度。

交叉熵 + softmax 合并梯度 = pred - y_true。
"""

import numpy as np


def cross_entropy_loss(pred, y_true):
    """交叉熵损失：pred 为 softmax 输出概率，y_true 为 one-hot 标签。"""
    eps = 1e-12
    return -np.sum(y_true * np.log(pred + eps))


def cross_entropy_grad(pred, y_true):
    """交叉熵 + softmax 合并后的梯度（对输出层 z 的梯度）：pred - y_true。"""
    return pred - y_true
