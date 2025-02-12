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

    def _elem_call(self, x):
        assert len(x) == len(self.w), f"Shape mismatch! ({len(x)} vs. {len(self.w)})"
        act = sum([wi*xi for wi, xi in zip(self.w, x)], self.b)
        out = act.tanh()
        return out

    def __call__(self, x):
        if isinstance(x, list) and isinstance(x[0], list):
            res = []
            for x_ in x:
                print("WTF")
                res.append(self._elem_call(x_))
            return res
        return self._elem_call(x)

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, number_input, number_outputs, seed=420):
        self.neurons = []
        print(f'New Layer: {number_input}->{number_outputs}')
        for _ in range(number_outputs):
            self.neurons.append(Neuron(number_input, seed=seed))

    def __call__(self, X):
        out = []
        for x in X:
            out.append([n(x) for n in self.neurons])
        return out

    def parameters(self):
        return list(itertools.chain.from_iterable([n.parameters() for n in self.neurons]))


class MLP(Module):
    def __init__(self, number_inputs, number_outputs, seed=420):
        sz = [number_inputs] + number_outputs
        self.layers = [Layer(sz[i], sz[i+1], seed=seed) for i in range(len(number_outputs))]

    def __call__(self, x):
        for idx, l in enumerate(self.layers):
            x = l(x)
            print(f'L{idx}: {x}')
        return x

    def parameters(self):
        return list(itertools.chain.from_iterable([l.parameters() for l in self.layers]))

