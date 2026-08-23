"""诊断监控职责：观察训练过程中各层的健康状态。

本模块只做『只读诊断』，不修改任何参数，也不参与前反向计算。
所有函数接收 layer 实例（或 + 一批输入 x_list），返回标量 / 比例 / dict。
"""

import math


def dead_neuron_ratio(layer):
    """诊断：统计本层 ReLU 死亡神经元（z<=0 被截断为 0）比例。

    若训练中该值长期接近 1.0，说明该层神经元几乎全部不激活、
    梯度无法回流，是训练停滞的典型原因。

    注意：这是针对【当前喂入的单个样本】的瞬时比例，不同样本会跳。
    要看整体/平均死亡率请用 dead_ratio_over_batch()。
    """
    if not layer.z or layer.is_output:
        return None
    dead = sum(1 for v in layer.z if v <= 0)
    return dead / len(layer.z)


def dead_ratio_over_batch(layer, x_list):
    """诊断：基于一批样本统计【整体/永久死亡】比例。

    做法：对每个神经元 i，统计它在 batch 里 'z_i <= 0'（即 ReLU 输出恒为 0）
    出现的次数；若某神经元在整批里一次都没激活（dead_count == len(batch)），
    则视为'永久死亡'。返回永久死亡神经元占比。

    这比死盯单个样本的 dead_neuron_ratio 更能反映网络真实状态：
    - 接近 0   ：神经元对这批输入基本都能响应（健康）
    - 接近 1.0 ：整层几乎对所有样本都不激活（梯度断流，训练停滞）
    """
    if layer.is_output:
        return None
    n = len(x_list)
    dead_count = [0] * layer.output_size
    for x in x_list:
        z = []
        for i in range(layer.output_size):
            z_row = 0.0
            for j in range(layer.input_size):
                z_row += layer.w[j][i] * x[j]
            z.append(z_row + layer.b[i])
        for i in range(layer.output_size):
            if z[i] <= 0:
                dead_count[i] += 1
    perm_dead = sum(1 for c in dead_count if c == n)
    return perm_dead / len(dead_count)


def weight_norm(layer):
    """诊断：权重矩阵 Frobenius 范数（衡量权重整体量级，爆炸/消失预警）。"""
    return math.sqrt(sum(w * w for col in layer.w for w in col))


def grad_norm(layer):
    """诊断：本 batch 累积梯度的 Frobenius 范数（梯度爆炸/消失预警）。"""
    return math.sqrt(sum(g * g for col in layer.grad_w for g in col))


def activation_mean(layer):
    """诊断：本层前向输出（激活后）的均值。

    隐藏层可观察 ReLU 是否长期输出接近 0（配合 dead_neuron_ratio）。
    """
    if not layer.z:
        return None
    if layer.is_output:                     # 输出层看 softmax 概率均值
        return sum(layer.z) / len(layer.z)
    act = layer.activation(layer.z)
    return sum(act) / len(act)


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
