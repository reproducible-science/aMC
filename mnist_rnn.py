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

#two-layer recurrent neural network for classifying two mnist digits
#at each site, show the net a binary vector, depending on whether pixel is black or white
#pass the final hidden state of the RNN through a fully-connect classifier layer

class RNNClassifier(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(RNNClassifier, self).__init__()
        self.rnn = torch.nn.GRU(input_size=2, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x):
        output, hidden = self.rnn(x)
        return self.linear(output[:,-1, :])


#initialize weights from gaussian distribution
def init_net(net, scale):
    for i, mod in enumerate(net.modules()):
        if isinstance(mod, torch.nn.Linear):
            torch.nn.init.normal_(mod.weight, std=scale)
            torch.nn.init.normal_(mod.bias, std=scale)


#download training and testing data, binarize and flatten

transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(torch.flatten),
    lambda x: x>.25,
    lambda x : torch.nn.functional.one_hot(x.to(torch.int64), num_classes = 2),
    lambda x: x.float()
    ])
training_dataset = torchvision.datasets.MNIST('data', train=True, download=True,
                    transform=transform)

testing_dataset = torchvision.datasets.MNIST('data', train=False, download = True,
                    transform=transform)


#only train on digits 0 and 1, move all data to device
train_loader = torch.utils.data.DataLoader(training_dataset,batch_size = 60000)
test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=10000)

for data,target in test_loader:
    test_data = data.to(device)[(target==0) | (target==1)]
    test_target = target.to(device)[(target==0) | (target==1)]

for data,target in train_loader:
    train_data = data.to(device)[(target==0) | (target==1)]
    train_target = target.to(device)[(target==0) | (target==1)]

#create a loader for mini-batch learning w/ batch size of 1500
new_training_dataset = torch.utils.data.TensorDataset(train_data, train_target)
train_loader = torch.utils.data.DataLoader(new_training_dataset,batch_size = 1500, shuffle = True)


#initialize neural network and optimizer

hidden_dim = 64

init_sigma = 1e-3
epsilon = 0

model = RNNClassifier(hidden_dim).to(device)
init_net(model, init_sigma)

optimizer = aMC(model.parameters(), init_sigma = init_sigma, epsilon = epsilon, minibatch=True)

#optimize net using aMC w/ mini-batching to classify MNIST digits
num_epochs = int(1e6)
train_stats = np.zeros((num_epochs, 3))

loss_function = torch.nn.CrossEntropyLoss()

for t in range(num_epochs):

    model.train()

    #track loss for whole epoch
    tot_loss = 0

    #update on mini-batches
    for i, (train_data, train_target) in enumerate(train_loader):
        def loss_closure():
            output = model(train_data)
            loss = loss_function(output, train_target)
            return loss

        #apply mutation to net, update net if move accepted
        loss = optimizer.step(loss_closure)                
        tot_loss += loss.item()
    
    train_stats[t,0] = tot_loss/(i+1)


    model.eval()
    output = model(test_data)
    loss = loss_function(output, test_target)                
    train_stats[t,1] = loss


    pred = torch.nn.functional.log_softmax(output).argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    acc = pred.eq(test_target.view_as(pred)).sum().item()
    train_stats[t,2] = acc/test_target.shape[0]
    print(train_stats[t])
