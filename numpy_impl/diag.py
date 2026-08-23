"""诊断监控职责（numpy 版）：观察训练过程中各层的健康状态。

本模块只做『只读诊断』，不修改任何参数。所有函数接收 layer 实例，
返回标量 / 比例 / dict。

注：dead_ratio_over_batch（基于整批样本统计）按要求保持骨架，未在此实现。
"""

import numpy as np


def dead_neuron_ratio(layer):
    """诊断：统计本层 ReLU 死亡神经元（z<=0 被截断为 0）比例。"""
    if layer.Z is None or len(layer.Z) == 0 or layer.is_output:
        return None
    z = np.asarray(layer.Z).ravel()
    dead = np.sum(z <= 0)
    return float(dead) / len(z)


def weight_norm(layer):
    """诊断：权重矩阵 Frobenius 范数（衡量权重整体量级）。"""
    return float(np.linalg.norm(layer.W))


def grad_norm(layer):
    """诊断：本 batch 累积梯度的 Frobenius 范数（梯度爆炸/消失预警）。"""
    return float(np.linalg.norm(layer.dW))


def activation_mean(layer):
    """诊断：本层前向输出（激活后）的均值。"""
    if layer.Z is None or len(layer.Z) == 0:
        return None
    z = np.asarray(layer.Z).ravel()
    if layer.is_output:                     # 输出层看 softmax 概率均值
        return float(np.mean(z))
    act = np.maximum(0, z)                  # 隐藏层看 ReLU 输出均值
    return float(np.mean(act))


def monitor(layer):
    """汇总本层监控指标，返回一个 dict（供 network 每轮打印）。"""
    info = {
        "w_norm": weight_norm(layer),
        "g_norm": grad_norm(layer),
        "act_mean": activation_mean(layer),
    }
    if not layer.is_output:
        info["dead"] = dead_neuron_ratio(layer)
    return info
