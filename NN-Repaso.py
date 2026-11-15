import math
import random
import numpy as np
import matplotlib.pyplot as plt
from data import Xtr, ytr   # votantes: (age01, income01, voted2019) -> turned out 0/1

random.seed(1337)
np.random.seed(1337)
print("Ready - pure-Python autograd lab, no GPU needed.")


##-------------------- Inicializar clases ----------------------------------------------------------------
class Value:
    """A scalar node in a computation graph (micrograd style).

    Records its data, a grad slot (filled in TOMORROW by backward()), its
    children, and the LOCAL derivative of this node w.r.t. each child.
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data              # El valor numérico real
        self.grad = 0                 # Gradiente (se llena en backward)
        self._children = children     # De dónde vino este valor
        self._local_grads = local_grads  # Derivadas locales


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out_data = self.data * other.data
        local_grads = (other.data, self.data)         
        return Value(out_data, (self, other), local_grads)

    def __pow__(self, other):
        out_data = self.data ** other
        local_grads = (other * self.data ** (other-1),) # other es un número escalar, no un Value, así que se accede directamente sin .data   
        return Value(out_data, (self,), local_grads) 

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):             return self * -1
    def __radd__(self, other):     return self + other
    def __sub__(self, other):      return self + (-other)
    def __rsub__(self, other):     return other + (-self)
    def __rmul__(self, other):     return self * other
    def __truediv__(self, other):  return self * other ** -1
    def __rtruediv__(self, other): return other * self ** -1
    def __repr__(self):            return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


        # ===================== NEW today: the backward() method =====================
    def backward(self):
        # 1) TOPOLOGICAL SORT: list every node so each comes AFTER its children.
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        # 2) SEED: the output's gradient with respect to itself is 1.
        self.grad = 1
        # 3) PROPAGATE: walk in REVERSE, pushing each node's grad to its children.
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

a = Value(2.0)
b = Value(3.0)
c = a + b
e  = Value(1.0).exp()
l  = Value(2.0).log()

print(l)


##-------------------- Grafico computacional ----------------------------------------------------------------

def trace(root):
    """Walk back through ._children, collecting all nodes and edges."""
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._children:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def _print_graph(node, depth=0):
    """Plain-text fallback: an indented tree with local grads on the edges."""
    print("    " * depth + f"Value(data={node.data:.4f}, grad={node.grad:.4f})")
    for child, lg in zip(node._children, node._local_grads):
        print("    " * depth + f"    \u2514\u2500 [local grad d(parent)/d(child) = {lg:.4f}]")
        _print_graph(child, depth + 1)


def draw_graph(root):
    """Draw the computation graph as a DAG (graphviz if available, else text)."""
    try:
        from graphviz import Digraph
    except Exception:
        _print_graph(root)
        return None
    nodes, edges = trace(root)
    dot = Digraph(graph_attr={"rankdir": "LR"})   # left-to-right, like the lecture
    for n in nodes:
        dot.node(name=str(id(n)),
                 label="{ data %.3f | grad %.3f }" % (n.data, n.grad),
                 shape="record")
    for child, parent in edges:
        i = parent._children.index(child)         # which child slot this is
        lg = parent._local_grads[i]
        dot.edge(str(id(child)), str(id(parent)), label=" %.3f" % lg)
    return dot

print("Helpers ready: trace(root) and draw_graph(root).")





### ---------------------- Full net ---------------------------------------------------------------------

class Neuron:
    """One neuron: output = activation(w . x + b)."""
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
    """A row of neurons, all reading the same input."""
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


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

### ---------------------- SGD Full update ---------------------------------------------------------------

def grad_check(f, *inputs, h=1e-6, tol=1e-4):
    """Compare Value.backward() gradients to a central finite difference.

    f      : a function of len(inputs) Value args, returning a single Value
    inputs : plain floats - the point at which to check the gradient
    """
    # --- analytic gradient (our autograd) ---
    args = [Value(xi) for xi in inputs]
    f(*args).backward()
    analytic = [arg.grad for arg in args]
    # --- numerical gradient (central difference) ---
    numeric = []
    for i in range(len(inputs)):
        up, dn = list(inputs), list(inputs)
        up[i] += h
        dn[i] -= h
        f_up = f(*[Value(t) for t in up]).data
        f_dn = f(*[Value(t) for t in dn]).data
        numeric.append((f_up - f_dn) / (2 * h))
    # --- compare ---
    ok = True
    for i, (an, nu) in enumerate(zip(analytic, numeric)):
        d = abs(an - nu)
        ok = ok and d < tol
        print(f"  input {i}:  autograd={an:+.6f}   finite-diff={nu:+.6f}   diff={d:.1e}")
    print("PASS\n" if ok else "FAIL\n")
    return ok


print("check 1:  f(a, b) = a*b + a**3 - b/a")
grad_check(lambda a, b: a * b + a ** 3 - b / a, 2.0, -3.0)

print("check 2:  f(a, b, c) = relu(a*b) + exp(c)")
grad_check(lambda a, b, c: (a * b).relu() + c.exp(), 1.5, -2.0, 0.5)


def sigmoid(v):

    return 1 / (1 + (-v).exp())

print("check 3:  f(a, b) = (sigmoid(a*b) - 1)**2")
grad_check(lambda a, b: (sigmoid(a * b) - 1) ** 2, 0.7, -1.3)



random.seed(1337)
model = MLP(3, [8, 1])
lr = 1.0
steps = 300
history = []

for step in range(steps):
    # 1) FORWARD + LOSS: mean squared error over all N voters.
    total = Value(0.0)
    for xi, yi in zip(Xtr, ytr):
        pred = sigmoid(model(xi))          # probability in (0, 1)
        total = total + (pred - yi) ** 2   # squared error for this voter
    loss = total * (1.0 / len(Xtr))        # average

    # 2) ZERO THE GRADS (params persist between steps - clear last step's grads!).
    for p in model.parameters():
        p.grad = 0

    # 3) BACKWARD: fill p.grad for every parameter.
    loss.backward()

    # 4) SGD UPDATE: step each parameter downhill.
    for p in model.parameters():
        p.data -= lr * p.grad

    history.append(loss.data)
    if step % 50 == 0 or step == steps - 1:
        print(f"step {step:3d}   loss {loss.data:.4f}")