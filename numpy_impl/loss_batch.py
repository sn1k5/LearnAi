"""批次版损失（loss_batch）：交叉熵，按 batch 平均（mean 约定）（numpy 版）。

与 loss.py 的差异（新增文件，二者可共存，不覆盖旧文件）：
  - 输入 pred/y_true 形状 [N, output_size]（行样本）。
  - loss 为各样本交叉熵的均值（标量）。

梯度约定：
  - cross_entropy_grad 返回 pred - y_true（未除 N），
    除以 N 的『平均』发生在 layer_batch.backward 内，两者相乘等价于对平均损失求梯。
"""

import numpy as np


def cross_entropy_loss(pred, y_true):
    """交叉熵损失（mean）：pred=[N, out]，y_true=[N, out] one-hot，返回标量=样本均值。"""
    eps = 1e-12
    # 各样本类内求和 -> 再对各样本取均值
    return -np.mean(np.sum(y_true * np.log(pred + eps), axis=1))


def cross_entropy_grad(pred, y_true):
    """交叉熵 + softmax 合并梯度（对输出层 z）：pred - y_true，形状 [N, out]。

    注意：此处不除以 N，N 的平均在 layer_batch.backward 内完成。
    """
    return pred - y_true