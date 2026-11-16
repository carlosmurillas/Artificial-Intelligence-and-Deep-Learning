import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Datos de votantes (age01, income01, voted2019) -> turned out 0/1
from data import Xtr, ytr


class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data ** other, (self,), (other * self.data ** (other - 1),))

    def log(self): return Value(math.log(self.data), (self,), (1 / self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other ** -1
    def __rtruediv__(self, other): return other * self ** -1
    def __repr__(self): return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

##-------------------- Perceptron and Layer ----------------------------------------------------------------

class Neuron:
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


def sigmoid(v):
    return 1 / (1 + (-v).exp())

##-------------------- Multi Layer Perceptron----------------------------------------------------------------


class MLP:
    """A stack of layers. The LAST layer is linear (nonlin=False)."""
    def __init__(self, nin, nouts): #nin = dimension, nouts =
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1], nonlin=(i != len(nouts) - 1))
                       for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

##-------------------- Backpropagation----------------------------------------------------------------


def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

##-------------------- Training loop ----------------------------------------------------------------

random.seed(1337)

model = MLP(3, [8, 1])
lr = 1.0
steps = 300
history = []

for step in range(steps):
    total = Value(0.0)
    for xi, yi in zip(Xtr, ytr):
        pred = sigmoid(model(xi))          # probability in (0, 1)
        total = total + (pred - yi) ** 2   # squared error for this voter
    loss = total * (1.0 / len(Xtr))        # average

    for p in model.parameters():
        p.grad = 0

    loss.backward()

    for p in model.parameters():
        p.data -= lr * p.grad

    history.append(loss.data)
    if step % 50 == 0 or step == steps - 1:
        print(f"step {step:3d}   loss {loss.data:.4f}")