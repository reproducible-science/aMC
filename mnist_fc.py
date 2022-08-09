import torch
import torchvision
import numpy as np
from aMC_optimizer import aMC


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("device: "+ device)


#won't need to calculate gradients anywhere, this speeds up computation
torch.autograd.set_grad_enabled(False)


#fully-connected neural network for classifying mnist
class amc_net(torch.nn.Module):
    def __init__(self, hidden_dim, num_hidden_layers, act= "tanh",layernorm = True):
        super().__init__()
        

        if act == "tanh":
            act_f = torch.nn.Tanh()
        else:
            act_f = torch.nn.ReLU()

        model = []

        #input layer
        if layernorm:
            model += [torch.nn.Linear(784, hidden_dim, bias = False), torch.nn.LayerNorm(hidden_dim), act_f]
        else:
            model += [torch.nn.Linear(784, hidden_dim), act_f]

        #hidden layers
        for i in range(num_hidden_layers):
            if layernorm:
                model += [torch.nn.Linear(hidden_dim, hidden_dim, bias = False), torch.nn.LayerNorm(hidden_dim), act_f]
            else:
                model += [torch.nn.Linear(hidden_dim, hidden_dim), act_f]

        #output layer
        model += [torch.nn.Linear(hidden_dim,10)]         

        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

#initialize weights from gaussian distribution
def init_net(net, scale, bias = False):
    for i, mod in enumerate(net.modules()):
        if isinstance(mod, torch.nn.Linear):
            torch.nn.init.normal_(mod.weight, std=scale)
            if bias:
                torch.nn.init.normal_(mod.bias, std=scale)


#download training and testing data, normalize and flatten
transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
    ,torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ,torchvision.transforms.Lambda(torch.flatten)

    ])
training_dataset = torchvision.datasets.MNIST('data', train=True, download=True,
                    transform=transform)

testing_dataset = torchvision.datasets.MNIST('data', train=False, download = True,
                    transform=transform)


#full-batch learning, move all data to device
train_loader = torch.utils.data.DataLoader(training_dataset,batch_size = 60000)
test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=10000)


for data,target in train_loader:
    train_data = data.to(device)
    train_target = target.to(device)

for data,target in test_loader:
    test_data = data.to(device)
    test_target = target.to(device)


#initialize neural network and optimizer

hidden_dim = 64
num_hidden_layers = 3

init_sigma = 1e-3
epsilon = 1e-3

model = amc_net(hidden_dim, num_hidden_layers).to(device)
init_net(model, init_sigma)

optimizer = aMC(model.parameters(), init_sigma = init_sigma, epsilon = epsilon)

#optimize net using aMC to classify MNIST digits
num_epochs = int(1e6)
train_stats = np.zeros((num_epochs, 3))

loss_function = torch.nn.CrossEntropyLoss()

for t in range(num_epochs):

    model.train()
    def loss_closure():
        output = model(train_data)
        loss = loss_function(output, train_target)
        return loss
    
    #apply mutation to net, update net if move accepted
    loss = optimizer.step(loss_closure)

    train_stats[t,0] = loss.item()


    #if model has been updated: calculate loss and accuracy on validation set
    if t ==0 or (t > 0 and train_stats[t, 0] != train_stats[t-1,0]):
        model.eval()
        output = model(test_data)
        loss = loss_function(output, test_target)                
        train_stats[t,1] = loss


        pred = torch.nn.functional.log_softmax(output).argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        acc = pred.eq(test_target.view_as(pred)).sum().item()
        train_stats[t,2] = acc/test_target.shape[0]
    else:
        train_stats[t,1] = train_stats[t-1,1]
        train_stats[t,2] = train_stats[t-1,2]

    print(train_stats[t])
