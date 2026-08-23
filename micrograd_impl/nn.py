"""神经网络组件（nn）：基于 value 自动微分引擎的 neuron/layer/mlp。

功能概览：
  - neuron：单个神经元，act(sum(wi*xi) + b)，激活支持 tanh/relu/linear。
  - layer：一层 nout 个 neuron，__call__ 返回输出列表。
  - mlp：多层感知机（中间层 tanh、最后一层 linear），提供 parameters()
    与 zero_grad()，配合 value.backward() 即可完成一次反向传播。

与现有实现的关系（新增文件，不覆盖旧实现）：
  - numpy_impl / list_impl 手写每层前向/反向；本模块前向即建图、反向全由
    engine.value 自动完成，可与手写版对照学习自动微分机制。

职责边界：
  - 只负责网络结构与前向建图；参数更新由外部训练循环完成
    （p.data -= lr * p.grad）。

运行方式：
  - cd LearnAi && python -c "from micrograd_impl.nn import mlp; \
    m=mlp(3,[4,4,1]); out=m([1.0,2.0,3.0]); loss=out[0]*out[0]; \
    loss.backward(); print(len(m.parameters()))"
  - 训练范式：每步先 zero_grad()，再 loss.backward()，然后逐参数更新。
"""

import random

from .engine import value


class neuron:
    """单个神经元：w·x + b 后按 activation 过激活（tanh/relu/linear）。"""

    def __init__(self, nin, activation='tanh'):
        # 权重 uniform(-1,1) 随机初始化，偏置从 0 出发（均为 value 节点）
        self.w = [value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = value(0.0)
        self.activation = activation

    def __call__(self, x):
        # 前向：act(sum(wi*xi) + b)；sum 初值取 b，w·x+b 全程在计算图内
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == 'tanh':
            return act.tanh()
        if self.activation == 'relu':
            return act.relu()
        if self.activation == 'linear':
            return act                      # linear 即恒等，不过激活
        raise ValueError(f"不支持的激活类型: {self.activation}")

    def parameters(self):
        """本神经元的全部可学习参数（权重 + 偏置）。"""
        return self.w + [self.b]


class layer:
    """一层神经元：nout 个 neuron 并联，__call__ 返回 nout 个 value 的列表。"""

    def __init__(self, nin, nout, activation='tanh'):
        self.neurons = [neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        # 各神经元共享同一输入 x，输出按顺序拼接为列表
        return [n(x) for n in self.neurons]

    def parameters(self):
        """汇总本层全部神经元的参数。"""
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params


class mlp:
    """多层感知机：nin -> nouts 逐层堆叠，中间层 tanh、最后一层 linear。"""

    def __init__(self, nin, nouts):
        sz = [nin] + list(nouts)
        self.layers = []
        for i in range(len(nouts)):
            # 最后一层 activation='linear'（原始得分），其余层 'tanh'
            act = 'linear' if i == len(nouts) - 1 else 'tanh'
            self.layers.append(layer(sz[i], sz[i + 1], activation=act))

    def __call__(self, x):
        # 输入可为 list[float] 或 list[value]，float 统一包装为 value 以入图
        x = [xi if isinstance(xi, value) else value(xi) for xi in x]
        for lay in self.layers:
            x = lay(x)                      # 逐层前向，上一层输出作下一层输入
        return x

    def parameters(self):
        """全部可学习参数（按层序展开）。"""
        params = []
        for lay in self.layers:
            params.extend(lay.parameters())
        return params

    def zero_grad(self):
        """清空全部参数梯度（训练循环每步前调用）。"""
        for p in self.parameters():
            p.grad = 0.0
