import numpy as np

from .optimizer import zero_grad as _zero_grad
from .optimizer import update as _update

class my_layer_batch:
    def __init__(self, input_size, output_size, is_output):
        self.input_size = input_size
        self.output_size = output_size
        self.is_output = is_output

        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.B = np.zeros(output_size)

        self.dW = np.zeros((input_size, output_size))
        self.dB = np.zeros(output_size)
        self.derivative_activation = None # 激活函数求导,用于方向传播Loss

        self.X = None
        self.Z = None

        self.vW = np.zeros((input_size, output_size))
        self.vB = np.zeros(output_size)

    def forward(self, X):
        self.X = X # 反向传播作为求dW的参数
        self.Z = X @ self.W + self.B
        
        if self.is_output:
            return self.softmax(self.Z)
        
        self.derivative_activation = self.Z >= 0 # derivative of ReLu
        return self.activation(self.Z) # ReLu

    def backward(self, L):
        if not self.is_output:
            L *= self.derivative_activation
        
        n = L.shape[0]
        self.dW = (self.X.T @ L) / n 
        self.dB = np.sum(L, axis=0) / n
        return L @ self.W.T
    
    def activation(self, Z):
       return np.maximum(0, Z)


    # ------------------------- 委托：参数更新 -------------------------
    def zero_grad(self):
        """清空累积梯度（保留接口，批次模式实际由 backward 覆盖写入）。"""
        _zero_grad(self)

    def update(self, lr, momentum=0.0):
        """SGD 更新（含 momentum），实现见 optimizer。"""
        _update(self, lr, momentum)



    # ------------------------- 输出层 softmax（沿类别方向 axis=1） -------------------------
    def softmax(self, Z):
        m = np.max(Z, axis=1, keepdims=True)
        e = np.exp(Z - m)
        return e / np.sum(e, axis=1, keepdims=True)


    