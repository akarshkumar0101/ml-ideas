import torch
from torch import nn
import torchvision

from tqdm import tqdm

import metrics

class MNIST:
    def __init__(self, bs_train=1000, bs_test=3000):
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))])

        self.ds_train = torchvision.datasets.MNIST('~/datasets/mnist', train=True,
                                                   download=True, transform=self.transform)
        self.ds_test = torchvision.datasets.MNIST('~/datasets/mnist', train=False,
                                                  download=True, transform=self.transform)
        self.bs_train, self.bs_test = bs_train, bs_test
        self.dl_train = torch.utils.data.DataLoader(self.ds_train,
                                                    batch_size=self.bs_train, shuffle=True)
        self.dl_test = torch.utils.data.DataLoader(self.ds_test,
                                                   batch_size=self.bs_test, shuffle=True)
        self.loss_fn = nn.CrossEntropyLoss()

    def load_all_data(self, device='cpu'):
        data = [(Xb, Yb) for Xb, Yb in self.loader_train]
        self.X_train = torch.cat([di[0] for di in data], dim=0).to(device)
        self.Y_train = torch.cat([di[1] for di in data], dim=0).to(device)
        
        data = [(Xb, Yb) for Xb, Yb in self.loader_test]
        self.X_test = torch.cat([di[0] for di in data], dim=0).to(device)
        self.Y_test = torch.cat([di[1] for di in data], dim=0).to(device)

    def calc_performance(self, net, dl=None, n_batches=None, tqdm=None, device=None):
        if dl is None:
            dl = self.dl_test
            
        net = net.to(device)
        
        meter_acc = metrics.AverageMeter()
        meter_loss = metrics.AverageMeter()
        
        loop = enumerate(dl)
        if tqdm is not None:
            loop = tqdm(loop, leave=False, total=len(dl) if n_batches is None else n_batches)
            
        for batch_idx, (x, y) in loop:
            x, y = x.to(device), y.to(device)
            if batch_idx==n_batches:
                break
            yp = net(x)
            loss = self.loss_fn(yp, y).item()
            acc = (yp.argmax(dim=-1)==y).sum().item()/len(x)
            
            meter_acc.update(acc, len(x))
            meter_loss.update(loss, len(x))
            
        return {'loss': meter_loss.avg, 'acc': meter_acc.avg,
                'meter_loss': meter_loss, 'meter_acc': meter_acc}

#     # TODO: do not use .item() anywhere and rather just accumulate gpu tensor data.
#     # transferring from gpu mem to cpu mem takes FOREVER in clock time
#     def calc_pheo_fitness(self, pheno, n_sample=5000, device='cpu', ds='train'):
#         if ds=='train':
#             X, Y = self.X_train, self.Y_train
#         else:
#             X, Y = self.X_test, self.Y_test
#         if n_sample is None:
#             idx = torch.arange(len(X))
#         else:
#             idx = torch.randperm(len(X))[:n_sample]
            
#         X_batch, Y_batch = X[idx].to(device), Y[idx].to(device)

#         Y_batch_pred = pheno(X_batch)
#         loss = self.loss_func(Y_batch_pred, Y_batch).item()
#         n_correct = (Y_batch_pred.argmax(dim=-1)==Y_batch).sum().item()
#         accuracy = n_correct/len(Y_batch)*100.
#         return {'fitness': -loss, 'loss': loss, 'accuracy': accuracy}


from torch import nn
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.MaxPool2d(3),
        )
        self.classification = nn.Sequential(
            nn.Linear(32, 10),
        )
    def forward(self, x):
        x = self.seq(x)
        x = x.reshape(len(x), -1)
        x = self.classification(x)
        return x
    
    
    
if __name__=='__main__':
    net = mnist.Network()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)

    meter_acc = metrics.AverageMeter(keep_verbose_stats=True)
    meter_loss = metrics.AverageMeter(keep_verbose_stats=True)

    for epoch_idx in range(5):
        for x, y in tqdm(ds.dl_train):
            yp = net(x)
            loss = ds.loss_fn(yp, y)
            acc = (yp.argmax(dim=-1)==y).sum().item()/len(x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            meter_acc.update(acc, len(x))
            meter_loss.update(loss.item(), len(x))
    plt.plot(np.array(meter_loss.data)[:, 0])