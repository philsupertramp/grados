import random
from grados.engine import Value
from grados.nn import Neuron, Layer, MLP

random.seed(420)

nn = Neuron(4)
layer = Layer(4, 2)
layer2 = Layer(4, 1)
inputs = [[1,2,3,4,], [2,3,4,5]]
print(nn(inputs))

print(layer(inputs))
print(layer2(inputs))

print("MLP!!!!!!!!!!!!!!!!")
mlp = MLP(4, [2, 3, 5, 1])
print("STARTING")

print(mlp(inputs))
#print(nn.parameters())
