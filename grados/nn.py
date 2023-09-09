import random
import itertools
from grados.engine import Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, number_inputs, seed=420):
        random.seed(seed)
        self.w = [Value(random.uniform(-1,1)) for _ in range(number_inputs)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        act = sum([wi*xi for wi, xi in zip(self.w, x)], self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, number_input, number_outputs, seed=420):
        self.neurons = [Neuron(number_input, seed=seed) for _ in range(number_outputs)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return list(itertools.chain.from_iterable([n.parameters() for n in self.neurons]))


class MLP(Module):
    def __init__(self, number_inputs, number_outputs, seed=420):
        sz = [number_inputs] + number_outputs
        self.layers = [Layer(sz[i], sz[i+1], seed=seed) for i in range(len(number_outputs))]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        return list(itertools.chain.from_iterable([l.parameters() for l in self.layers]))

