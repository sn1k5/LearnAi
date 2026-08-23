"""参数更新职责：梯度清零与 SGD(+momentum) 权重更新。

本模块只负责『拿累积好的梯度去更新参数』，不含前向/反向计算，
也不含诊断监控。函数接收 layer 实例，直接修改其 w/b/v_w/v_b。
"""

import math


def zero_grad(layer):
    """清空累积梯度，每个 mini-batch 开始前调用。"""
    layer.grad_w = [[0.0 for _ in range(layer.output_size)]
                    for _ in range(layer.input_size)]
    layer.grad_b = [0.0 for _ in range(layer.output_size)]


def update(layer, lr, momentum=0.0):
    """SGD 更新；momentum>0 时按 v = β·v - η·∇ 累加速度后更新权重。

    公式: v_{t+1} = β·v_t - η·∇W ;  W_{t+1} = W + v_{t+1}
    """
    for i in range(layer.output_size):
        for j in range(layer.input_size):
            layer.v_w[j][i] = momentum * layer.v_w[j][i] - lr * layer.grad_w[j][i]
            layer.w[j][i] += layer.v_w[j][i]
        layer.v_b[i] = momentum * layer.v_b[i] - lr * layer.grad_b[i]
        layer.b[i] += layer.v_b[i]
