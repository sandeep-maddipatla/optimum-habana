import os

# Use the eager mode
os.environ['PT_HPU_LAZY_MODE'] = '0'

# Verify the environment variable is set
print(f"PT_HPU_LAZY_MODE: {os.environ['PT_HPU_LAZY_MODE']}")

import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

import habana_frameworks.torch.core as htcore


device = torch.device("hpu")

# Node features matrix
x_0 = torch.rand(50, 32, device=device)
adj_0 = torch.rand(50,50, device=device).round().long()
identity = torch.eye(50, device=device)
adj_0 = adj_0 + identity

n_clusters_0 = 50
n_clusters_1 = 5

w_gnn_emb = torch.rand(32, 16, device=device)
w_gnn_pool = torch.rand(32, n_clusters_1, device=device)

z_0 = torch.relu(adj_0 @ x_0 @ w_gnn_emb)
s_0 = torch.softmax(torch.relu(adj_0 @ x_0 @ w_gnn_pool), dim=1)

x_1 = s_0.t() @ z_0
adj_1 = s_0.t() @ adj_0 @ s_0

import os.path as osp
from math import ceil

import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseGCNConv as GCNConv
from torch_geometric.nn import dense_diff_pool

max_nodes = 150


class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes



dataset = TUDataset('data', name='PROTEINS', transform=T.ToDense(max_nodes),
                    pre_filter=MyFilter())

                    
dataset = dataset.shuffle()
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DenseDataLoader(test_dataset, batch_size=32)
val_loader = DenseDataLoader(val_dataset, batch_size=32)
train_loader = DenseDataLoader(train_dataset, batch_size=32)


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super(GNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))


    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](x, adj, mask))
            x = x.permute(0, 2, 1)
            x = self.bns[step](x)
            x = x.permute(0, 2, 1)
        return x


class DiffPool(torch.nn.Module):
    def __init__(self):
        super(DiffPool, self).__init__()

        num_nodes = ceil(0.25 * max_nodes)        # 0.25*150 -> 38
        self.gnn1_pool = GNN(dataset.num_features, 64, num_nodes)
        self.gnn1_embed = GNN(dataset.num_features, 64, 64)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(64, 64, num_nodes)
        self.gnn2_embed = GNN(64, 64, 64, lin=False)

        self.gnn3_embed = GNN(64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(64, 64)
        self.lin2 = torch.nn.Linear(64, dataset.num_classes)

        self.lin_dbg1     = torch.nn.Linear(in_features=38, out_features=10)
        self.lin_dbg2_flt = torch.nn.Linear(in_features=38*38, out_features=2)
        self.lin_dbg3_flt = torch.nn.Linear(in_features=10*64, out_features=2)
        

    def forward(self, x, adj, mask=None):
    
        x      = torch.rand(32, 150, 64).to('hpu')
        adj    = torch.rand(32, 150, 150).to('hpu')
        s      = torch.load('./in_ten_s.pt').   to('hpu')
        #s      = torch.rand(32, 150, 38).to('hpu')       #if we generate this tensor, then GC passes

        mask_f = torch.rand(32, 150)
        mask   = (mask_f > 0.5).to('hpu')

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        x   = torch.randn(32, 38, 64).to('hpu')
        s   = torch.randn(32, 38, 10).to('hpu')

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x_flattened = x.view(32, -1) 
        ret = self.lin_dbg3_flt(x_flattened)
        
        return ret, l1 + l2, e1 + e2


model = DiffPool().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        output, _, _ = model(data.x, data.adj, data.mask)

        data_view_rnd = torch.randint(0, 100, (32,), dtype=torch.int64).to('hpu')
        
        loss = F.nll_loss(output, data_view_rnd)

        loss.backward()
        
        loss_all += loss.item()
        
        optimizer.step()
        
    return loss_all / len(train_dataset)



model.train()

model = torch.compile(model, backend="hpu_backend")

from tqdm.auto import trange

best_val_acc = test_acc = 0
for epoch in trange(1, 2):
    train_loss = train(epoch)



