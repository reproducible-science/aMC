# Training neural networks using Metropolis Monte Carlo and an adaptive variant

This repository contains a pytorch implementation of the aMC optimizer as described in https://arxiv.org/abs/2205.07408, and tutorials on how it can be used to optimize neural networks on the MNIST data set.

## General use

Any gradient-based optimizer can easily be replaced with the aMC optimizer. If the gradient-based optimizer is used as follows

```
optimizer = torch.optim.Adam(net.parameters(), lr = ...)

optimizer.zero_grad()
loss = loss_function(net(data),...)
loss.backward()
optimizer.step()
```
then the aMC optimizer can be used by defining a "closure" function which calculates the loss:

```
optimizer = aMC(net.parameters(), init_sigma = ...)

def loss_closure(data):
    loss = loss_function(net(data),...)
    return loss

optimizer.step(loss_closure)
```

while the rest of the training loop remains unchanged. Examples can be found in the tutorial files.
    
