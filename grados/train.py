from grados.nn import MLP

xs = [
    [[2,3,-1]],
    [[3,-1,0.5]],
    [[.5,1,1]],
    [[1,1,-1]]
]
ys = [1,-1, -1, 1]
n = MLP(3, [4,4,1])
EPOCHS = 500
learning_rate = 0.01
for epoch in range(EPOCHS):
    # forward pass
    ypred = [n(x) for x in xs]
    losses = [(yout[0][0] - ygt) ** 2 for ygt, yout in zip(ys, ypred)]
    loss = sum(losses)

    # backward pass
    n.zero_grad()
    loss.backward()

    # update
    lr = 1.0 - (1.0 - learning_rate)*epoch/EPOCHS
    for p in n.parameters():
        p.data += -lr * p.grad
        print(f'{p.grad}, ', end='')
    
    print()
    print(f'EPOCH {epoch}: loss = {loss}, lr = {lr}, losses = {losses}')

print([n(x) for x in xs])
