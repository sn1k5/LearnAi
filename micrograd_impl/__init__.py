"""micrograd_impl：micrograd 风格标量自动微分引擎与神经网络组件（新增模块）。

导出：value（标量自动微分节点）与 neuron/layer/mlp（基于 value 的网络组件）。
为「从零实现大模型」学习项目的 autograd 阶段，不依赖、也不修改
numpy_impl / list_impl 的任何现有实现。
"""

from .engine import value
from .nn import layer, mlp, neuron

__all__ = ["value", "neuron", "layer", "mlp"]
