import math

eps = 1e-12  # 防止 log(0)


def cross_entropy_loss(pred, y_true):
    """交叉熵损失：pred 为 softmax 输出概率，y_true 为 one-hot 标签"""
    return -sum(y_true[i] * math.log(pred[i] + eps) for i in range(len(y_true)))


def cross_entropy_grad(pred, y_true):
    """交叉熵 + softmax 合并后的梯度（对输出层 z 的梯度）：pred - y_true"""
    return [pred[i] - y_true[i] for i in range(len(y_true))]
