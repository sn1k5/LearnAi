import math
from .layer import layer
from .loss import cross_entropy_grad, cross_entropy_loss

class network:
    def __init__(self, input_size, hidden_dim, num_hidden_layers, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.layers = []
        prev_size = input_size
        for _ in range(num_hidden_layers):
            self.layers.append(layer(prev_size, hidden_dim))
            prev_size = hidden_dim
        self.layers.append(layer(hidden_dim, output_size, is_output=True))

    def predict(self, x):
        out = x
        for l in self.layers:
            out = l.forward(out)
        return out

    def train_step(self, x, y_true, lr = 0.1, update = True):
        pred = self.predict(x)
        loss = cross_entropy_loss(pred, y_true)
        grad_x = cross_entropy_grad(pred, y_true)
        for l in reversed(self.layers):
            grad_x = l.backward(grad_x, accumulate = not update)
        if update:
            for l in self.layers:
                l.update(lr)
        return loss

    def train(self, x_data, y_data, epochs, batch_size = 1, 
                lr_decay = 1.0, lr = 0.1):
        n = len(x_data)
        if batch_size > n:
            raise ValueError(f"batch_size ({batch_size}) cannot be greater than n ({n})")
        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            i = 0
            while i < n:
                bs = min(batch_size, n - i)
                batch_x = x_data[i:i + bs]
                batch_y = y_data[i:i + bs]
                for l in self.layers:
                    l.zero_grad()
                batch_loss = 0.0
                for k in range(bs):
                    batch_loss += self.train_step(batch_x[k], batch_y[k], lr, update=False)
                for l in self.layers:
                    l.update(lr / bs)
                total_loss += batch_loss
                i += bs
            history.append(total_loss / n)
            lr *= lr_decay
        return history
