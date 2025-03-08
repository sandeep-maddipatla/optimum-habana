import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import random
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.random as htrandom
def is_lazy():
    return os.getenv("PT_HPU_LAZY_MODE", "1") != "0"
################ Define Models ################
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2), 
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2), 
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1600, 10)
        )
    def forward(self, x):
        x = self.layers(x)
        return x
class CNN_BN(nn.Module):
    def __init__(self):
        super(CNN_BN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2), 
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2), 
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1600, 10)
        )
    def forward(self, x):
        x = self.layers(x)
        return x
class MLP_BN(nn.Module):
    def __init__(self):
        super(MLP_BN, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 10)
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c*h*w)
        return self.fc2(self.activation1(self.bn1(self.fc1(x))))
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 1024)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 10)
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c*h*w)
        return self.fc2(self.activation1(self.fc1(x)))
################################################
def run():
    device = "hpu"
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, drop_last=False)
    # model = CNN().to(device)
    # model = MLP().to(device)
    # model = MLP_BN().to(device)
    model = CNN_BN().to(device)
    print('Is Lazy:', is_lazy())
    if not is_lazy():
        model = torch.compile(model, backend="hpu_backend")
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0, momentum=0.9)
    for inputs, targets in tqdm(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        htcore.mark_step()
        optimizer.step()
        htcore.mark_step()
    os.makedirs('./save', exist_ok=True)
    filename = os.path.join('./save', 'last_ckpt.pth')
    torch.save({
            'state_dict': model.state_dict(),
        }, filename)
    print('--------------Train Finished--------------')
if __name__=='__main__':
    # set seeds
    torch.set_num_threads(1)
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    state = htrandom.get_rng_state()
    htrandom.set_rng_state(state)
    initial_seed = htrandom.initial_seed()
    htrandom.manual_seed(seed)
    run()