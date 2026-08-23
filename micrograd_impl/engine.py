"""标量自动微分引擎（value）：micrograd 风格的标量反向传播（纯 Python）。

功能概览：
  - value 是标量计算图节点：data 存前向值，grad 存反向梯度（初始 0.0）。
  - 算术运算符（+ - * / ** 与 neg）和超越函数（tanh/relu/exp/log）均生成
    新节点，并通过 _backward 闭包登记局部梯度回传公式，自动构建计算图。
  - backward() 构建拓扑序后逆序回传，一次调用完成整张子图的反向传播。

与现有实现的关系（新增文件，不覆盖旧实现）：
  - numpy_impl / list_impl 是手写前向/反向（每层显式 dW/dB）；本模块改为
    由计算图自动微分，二者可共存对照学习：手写版理解公式，autograd 版
    理解『记录运算 -> 逆拓扑序回传』的通用机制。

职责边界：
  - 只负责标量级『记录运算 + 反向传梯度』，不含网络结构与参数更新
    （见同目录 nn.py 的 neuron/layer/mlp）。
  - 不依赖 numpy，纯 Python 标量运算。

运行方式：
  - cd LearnAi && python -c "from micrograd_impl.engine import value; \
    a=value(2.0); b=value(-3.0); c=a*b+3; c.backward(); print(a.grad, b.grad)"
"""

import math


class value:
    """标量自动微分节点：data=前向值，grad=反向梯度，_prev=子节点，_op=运算标签。"""

    def __init__(self, data, _prev=(), _op=""):
        self.data = data
        self.grad = 0.0                     # 梯度初始 0，反向传播时逐节点累加
        self._backward = lambda: None       # 默认 no-op 闭包，由各运算注册真实回传
        self._prev = set(_prev)             # 子节点集合（计算图下游）
        self._op = _op                      # 产生本节点的运算标签（调试用）

    def __repr__(self):
        return f"value(data={self.data}, grad={self.grad})"

    # ------------------------- 辅助：拓扑序 -------------------------
    def _topo(self):
        """DFS + visited 收集从自身可达的全部节点，返回拓扑序（子在前、自身在后）。"""
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        return topo

    # ------------------------- 算术运算符 -------------------------
    def __add__(self, other):
        # 常数（int/float）包装成常数 value 节点，统一进入计算图
        other = other if isinstance(other, value) else value(other)
        out = value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = d(a+b)/db = 1，两个子节点都累加上游梯度
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        # other + self：加法交换律，等价 self + other
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, value) else value(other)
        out = value(self.data * other.data, (self, other), "*")

        def _backward():
            # 乘法法则：da 端梯度 = b.data * out.grad；db 端梯度 = a.data * out.grad
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        # other * self：乘法交换律，等价 self * other
        return self * other

    def __pow__(self, exp):
        # 幂运算仅支持 int/float 常数指数（指数不作为节点参与求导）
        assert isinstance(exp, (int, float)), "仅支持 int/float 指数"
        out = value(self.data ** exp, (self,), f"**{exp}")

        def _backward():
            # 幂法则：d(a^p)/da = p * a^(p-1)
            self.grad += exp * (self.data ** (exp - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        # -a = a * (-1)，复用乘法梯度规则
        return self * -1.0

    def __sub__(self, other):
        # a - b = a + (-b)
        return self + (-other)

    def __rsub__(self, other):
        # other - self = other + (-self)，常数先包装为 value
        other = other if isinstance(other, value) else value(other)
        return other + (-self)

    def __truediv__(self, other):
        # a / b = a * b^(-1)，复用乘法与幂的梯度规则
        return self * other ** -1.0

    def __rtruediv__(self, other):
        # other / self = other * self^(-1)
        other = other if isinstance(other, value) else value(other)
        return other * self ** -1.0

    # ------------------------- 超越函数 -------------------------
    def tanh(self):
        t = math.tanh(self.data)
        out = value(t, (self,), "tanh")

        def _backward():
            # tanh 导数：1 - tanh(x)^2 = 1 - t^2
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            # relu 导数：x>0 取 1，否则取 0（x=0 处约定取 0）
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = value(e, (self,), "exp")

        def _backward():
            # exp 导数即自身：d exp(x)/dx = exp(x)
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = value(math.log(self.data), (self,), "log")

        def _backward():
            # 自然对数导数：d ln(x)/dx = 1/x
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    # ------------------------- 反向传播 -------------------------
    def zero_grad(self):
        """将自身可达子图内所有节点的 grad 置 0（供训练循环每步前调用）。"""
        for v in self._topo():
            v.grad = 0.0

    def backward(self):
        """从本节点出发反向传播：清零可达子图 -> 置自身 grad=1.0 -> 逆拓扑序回传。

        梯度复位针对从该节点可达的整个子图（含全部中间节点），
        因此重复调用、或在旧计算图上追加新节点再调用，都不会残留旧梯度。
        """
        topo = self._topo()
        for v in topo:
            v.grad = 0.0
        self.grad = 1.0                      # 种子梯度：dL/dL = 1
        for v in reversed(topo):             # 逆拓扑序：自身最先回传局部梯度
            v._backward()
