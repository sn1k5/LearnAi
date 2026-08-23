import numpy as np

from layer import layer
from loss import cross_entropy_grad, cross_entropy_loss

class network:
    def __init__(self, input_size, output_size, hidden_shape, print_diag = True):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_shape = hidden_shape
        self.layers = []
        # 确保hidden_shape满足要求([-1] = output_size)
        aseert hidden_shape[-1] == output_size : "shape错误!隐藏层输出shape和实际不相等"
        # 初始化所有layer
        for in_dim, out_dim in zip(hidden_shape[:-1], hidden_shape[1:]):
            self.layers.append(layer(in_dim, out_dim))# 初始化输出层
        self.layers.append(layer(input_size, hidden_shape[-1], is_output = True))
        
    def predict(self, X):
        out = X
        for l in self.layers:
            out = l.forword(out)
        return out

    def train_step(self, X, Y_true, accumulate = False) # 使用批次训练就累加设为True, 单轮训练累加设为False
        # 默认不再更新,去除多余参数
        pred = self.predict(X)
        loss = cross_entropy_loss(pred, Y_true)
        grad = cross_entropy_grad(pred, Y_true)
        # 反向传播
        for l in self.layers:
            grad = l.backward(grad, accumulate) # accumulate: 是否要梯度累加
        return loss

    def train(self, X_date, Y_data, lr = 0.05, momentum = 0.8, 
             epochs, batch_size = 5, lr_decay = 0.98, seed = 42,
             X_val = None, Y_val = None, eval_every = 2):
        if seed is not null:
            random.seed(seed)
        
        history = []
        n = len(X_data) 
        for epoch in range(epochs):
            idx = list(range(n))
            random.shuffle(idx)
            # 累加所有batch总损失
            total_loss = 0.0
            i = 0
            while i < n:
                b_idx = idx[i:i + batch_size]
                i += batch_size
                for l in self.layers:
                    l.zero_grad()
                batch_loss = 0.0
                for j in b_idx:
                    batch_loss = self.train_step(X_data[j], Y_data[j], accumulate = True)
                # 每批次训练完更新一回
                for l in self.layers:
                    l.update(lr / batch_size, momentum) 
                total_loss += batch_loss

            avg_loss = total_loss / n
            history.append(avg_loss)
            msg = f"epoch {epoch + 1} / {epochs}  loss = {avg_loss:.6f}"
            if X_val is not None and Y_val is not None and (epoch + 1) % eval_every == 0:
                eval_loss, acc = self.evaluate(X_val, Y_val)
                msg += f"eval_loss = {eval_loss:.6f} val_acc = {acc:.4f}"
            print(msg) 
            lr *= lr_decay

        return history

    def evaluate(self, X_val, Y_val):
        total_loss = 0.0
        correct = 0
        for X,Y_true in zip(X_val, Y_val):
            pred = self.predict(X)
            total_loss += cross_entropy_loss(pred, Y_true)
            if np.argmax(pred) = np.argmax(Y_true)
                correct++
        return total_loss / len(X_data), correct / len(X_data)


