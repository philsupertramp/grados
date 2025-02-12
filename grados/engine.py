import math


def cast_other_value(fun):
    def inner(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return fun(self, other)
    return inner


class Value:
    def __init__(self, data, _children = (), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, op={self._op}, label={self.label}, child={self._prev})"

    @cast_other_value
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            #print(f'ADD: {self.grad} ->', end='')
            self.grad += 1.0 * out.grad
            #print(f' {self.grad}; {other.grad} -> ', end='')
            other.grad += 1.0 * out.grad
            #print(f'{other.grad} ', end='')
        
        out._backward = _backward
        return out

    @cast_other_value
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            #print(f'MM: {out.grad}; {self.grad} ->', end='')
            self.grad += other.data * out.grad
            #print(f' {self.grad}; {other.grad} -> ', end='')
            other.grad += self.data * out.grad
            #print(f'{other.grad} ', end='')

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-1. * other)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), f'Unsupported data type {type(other)}'
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += other * self.data ** (other - 1.) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x)+1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1.0 - t**2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    build_topo(c)

                topo.append(v)
                #print(f'Added {v}')
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            #print(f'Running for {node}: {node.grad} -> ', end='')
            node._backward()
            #print(f'{node.grad}')

