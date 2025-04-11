import os

# Use the eager mode
os.environ['PT_HPU_LAZY_MODE'] = '0'

# Verify the environment variable is set
print(f"PT_HPU_LAZY_MODE: {os.environ['PT_HPU_LAZY_MODE']}")

import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

import habana_frameworks.torch.core as htcore

# use rich traceback

from rich import traceback
traceback.install()

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

print(x_1.shape)
print(adj_1.shape)

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

        # print(f"X: {x.shape}")
        # print(f"ADJ: {adj.shape}")
        # print(f"MASK: {mask.shape}")
        # X: torch.Size([32, 150, 3])
        # ADJ: torch.Size([32, 150, 150])
        # MASK: torch.Size([32, 150])
        for step in range(len(self.convs)):
            # print(f"Step {step}")
            # print(self.convs[step])
            x = F.relu(self.convs[step](x, adj, mask))
            # print(f"after conv, x: {x.shape}")
            # print(self.bns[step])
            x = x.permute(0, 2, 1)
            x = self.bns[step](x)
            x = x.permute(0, 2, 1)
            # print(f"after bns, x: {x.shape}")
        return x


class DiffPool(torch.nn.Module):
    def __init__(self):
        super(DiffPool, self).__init__()

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(dataset.num_features, 64, num_nodes)
        self.gnn1_embed = GNN(dataset.num_features, 64, 64)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(64, 64, num_nodes)
        self.gnn2_embed = GNN(64, 64, 64, lin=False)

        self.gnn3_embed = GNN(64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(64, 64)
        self.lin2 = torch.nn.Linear(64, dataset.num_classes)

    def forward(self, x, adj, mask=None):
        # print(f"X: {x.shape}")
        # print(f"ADJ: {adj.shape}")
        # print(f"MASK: {mask.shape}")
        # X: torch.Size([32, 150, 3])
        # ADJ: torch.Size([32, 150, 150])
        # MASK: torch.Size([32, 150])
        # print(self.gnn1_pool)
        s = self.gnn1_pool(x, adj, mask)
        # print(f"S: {s.shape}")

        x = self.gnn1_embed(x, adj, mask)
        # print(f"x: {s.shape}")

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        # print(f"x: {x.shape}")
        # print(f"adj: {adj.shape}")
        # print(f"l1: {l1.shape}")
        # print(f"e1: {e1.shape}")

        #x_1 = s_0.t() @ z_0
        #adj_1 = s_0.t() @ adj_0 @ s_0

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DiffPool().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, _, _ = model(data.x, data.adj, data.mask)
        loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred, _, _ = model(data.x, data.adj, data.mask)
        print(pred)
        pred = pred.max(dim=1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)

model.train()

model = torch.compile(model, backend="hpu_backend")

from tqdm.auto import trange

best_val_acc = test_acc = 0
for epoch in trange(1, 151):
    train_loss = train(epoch)
    val_acc = test(val_loader)
    if val_acc > best_val_acc:
        test_acc = test(test_loader)
        best_val_acc = val_acc
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
          f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')