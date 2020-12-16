import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import math
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid,PPI
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, GCNConv, GAE, VGAE
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch.distributions import Normal
from torch import nn
from .layers import MetaGCNConv, MetaGatedGraphConv, MetaGRUCell, MetaGatedGCNConv
import torch.nn.functional as F
from utils.utils import uniform
from utils.edge_drop import EdgeDrop, DropMode
import ipdb
import numpy as np
from .layers import DGConv2d, DGConv1d



def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


############################### DGCNN architechture -- Encoder ########################################################################################
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

class DGCNN(torch.nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k


        self.LReLU = nn.LeakyReLU(negative_slope=0.2)

        self.conv1 = DGConv2d(120, 64, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)


        self.conv2 = DGConv2d(64 , 64, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = DGConv2d(64 , 128, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = DGConv2d(128, 256, kernel_size=1, bias=True)
        self.bn4 = nn.BatchNorm2d(256)


        self.conv5 = DGConv1d(512, 256, kernel_size=1, bias=True)
        self.bn5 = nn.BatchNorm1d(256)


        self.conv6 = DGConv1d(256, args.emb_dims, kernel_size=1, bias=True)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)




    def forward(self, x, weights):

        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.LReLU(self.bn1(self.conv1(x, weights['encoder.conv1.weight'], weights['encoder.conv1.bias'])))
        x1 = x.max(dim=-1, keepdim=False)[0]

        # x = get_graph_feature(x1, k=self.k)
        x = self.LReLU(self.bn2(self.conv2(x, weights['encoder.conv2.weight'], weights['encoder.conv2.bias'])))
        x2 = x.max(dim=-1, keepdim=False)[0]

        # x = get_graph_feature(x2, k=self.k)
        x = self.LReLU(self.bn3(self.conv3(x, weights['encoder.conv3.weight'],weights['encoder.conv3.bias'])))
        x3 = x.max(dim=-1, keepdim=False)[0]

        # x = get_graph_feature(x3, k=self.k)
        x = self.LReLU(self.bn4(self.conv4(x, weights['encoder.conv4.weight'], weights['encoder.conv4.bias'])))
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.LReLU(self.bn5(self.conv5(x, weights['encoder.conv5.weight'], weights['encoder.conv5.bias'])))
        x = self.LReLU(self.bn6(self.conv6(x, weights['encoder.conv6.weight'], weights['encoder.conv6.bias'])))

        x = x.permute(0, 2, 1)

        # x = F.leaky_relu(self.linear1(x), negative_slope=0.2)
        # x = F.leaky_relu(self.linear2(x), negative_slope=0.2)
        return x
############################### DGCNN architechture -- Encoder ###########################################################################################


class Encoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.args = args
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
        if args.model in ['GAE']:
            self.conv2 = GCNConv(2 * out_channels, out_channels, cached=False)
        elif args.model in ['VGAE']:
            self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=False)
            self.conv_logvar = GCNConv(
                2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        if self.args.model in ['GAE']:
            return self.conv2(x, edge_index)
        elif self.args.model in ['VGAE']:
            return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

class MetaMLPEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(MetaMLPEncoder, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(in_channels, 2 * out_channels, bias=True)
        if args.model in ['GAE']:
            self.fc_mu = nn.Linear(2 * out_channels, out_channels, bias=True)
        elif args.model in ['VGAE']:
            self.fc_mu = nn.Linear(2 * out_channels, out_channels, bias=True)
            self.fc_logvar = nn.Linear(2 * out_channels, out_channels, bias=True)

    def forward(self, x, edge_index, weights, inner_loop=True):
        x = F.relu(F.linear(x, weights['encoder.fc1.weight'],weights['encoder.fc1.bias']))
        if self.args.model in ['GAE']:
            return F.relu(F.linear(x, weights['encoder.fc_mu.weight'],weights['encoder.fc_mu.bias']))
        elif self.args.model in ['VGAE']:
            return F.relu(F.linear(x,weights['encoder.fc_mu.weight'],\
                    weights['encoder.fc_mu.bias'])),F.relu(F.linear(x,\
                    weights['encoder.fc_logvar.weight'],weights['encoder.fc_logvar.bias']))

class MLPEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(MLPEncoder, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(in_channels, 2 * out_channels, bias=True)
        self.fc2 = nn.Linear(2 * out_channels, out_channels, bias=True)

    def forward(self, x, edge_index):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class GraphSignature(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(GraphSignature, self).__init__()
        self.args = args
        if self.args.use_gcn_sig:
            self.conv1 = MetaGCNConv(in_channels, 2*out_channels, cached=False)
            self.fc1 = nn.Linear(2*out_channels, 2*out_channels, bias=True)
            self.fc2 = nn.Linear(2*out_channels, 2*out_channels, bias=True)
            self.fc3 = nn.Linear(2*out_channels, out_channels, bias=True)
            self.fc4 = nn.Linear(2*out_channels, out_channels, bias=True)
        else:
            self.gated_conv1 = MetaGatedGraphConv(in_channels, args.num_gated_layers)
            self.fc1 = nn.Linear(in_channels, 2 * out_channels, bias=True)
            self.fc2 = nn.Linear(in_channels, 2 * out_channels, bias=True)
            self.fc3 = nn.Linear(in_channels, out_channels, bias=True)
            self.fc4 = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, edge_index, weights, keys):
        if self.args.use_gcn_sig:
            x = F.relu(self.conv1(x, edge_index, \
                    weights['encoder.signature.conv1.weight'],\
                    weights['encoder.signature.conv1.bias']))
        else:
            x = F.relu(self.gated_conv1(x, edge_index, weights,keys))

        x = x.sum(0)
        x_gamma_1 = F.linear(x, weights['encoder.signature.fc1.weight'],\
                weights['encoder.signature.fc1.bias'])
        x_beta_1 = F.linear(x, weights['encoder.signature.fc2.weight'],\
                weights['encoder.signature.fc2.bias'])
        x_gamma_2 = F.linear(x, weights['encoder.signature.fc3.weight'],\
                weights['encoder.signature.fc3.bias'])
        x_beta_2 = F.linear(x, weights['encoder.signature.fc4.weight'],\
                weights['encoder.signature.fc4.bias'])
        return torch.tanh(x_gamma_1), torch.tanh(x_beta_1),\
                torch.tanh(x_gamma_2), torch.tanh(x_beta_2)

class MetaSignatureEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(MetaSignatureEncoder, self).__init__()
        self.args = args
        self.conv1 = MetaGCNConv(in_channels, 2 * out_channels, cached=False)
        if args.model in ['GAE']:
            self.conv2 = MetaGCNConv(2 * out_channels, out_channels, cached=False)
        elif args.model in ['VGAE']:
            self.conv_mu = MetaGCNConv(2 * out_channels, out_channels, cached=False)
            self.conv_logvar = MetaGCNConv(
                2 * out_channels, out_channels, cached=False)
        # in_channels is the input feature dim
        self.signature = GraphSignature(args, in_channels, out_channels)
        if self.args.drop_edges:
            if self.args.drop_mode == 'equal':
                self.edge_drop = EdgeDrop(keep_prob=args.keep_prob, mode=DropMode.EQUAL, plot=False)
            else:
                self.edge_drop = EdgeDrop(keep_prob=args.keep_prob, mode=DropMode.WEIGHTED, plot=False)


    def forward(self, x, edge_index, weights, only_gae=False,  inner_loop=True, train=False, no_sig=False):
        if self.args.drop_edges and train:
            edge_index = self.edge_drop(edge_index)
        keys = list(weights.keys())
        sig_keys = [key for key in keys if 'signature' in key]
        if inner_loop:
            with torch.no_grad():
                sig_gamma_1, sig_beta_1, sig_gamma_2, sig_beta_2 = self.signature(x, edge_index, weights, sig_keys)
                self.cache_sig_out = [sig_gamma_1,sig_beta_1,sig_gamma_2,sig_beta_2]
        else:
            sig_gamma_1, sig_beta_1, sig_gamma_2, sig_beta_2 = self.signature(x, edge_index, weights, sig_keys)
            self.cache_sig_out = [sig_gamma_1,sig_beta_1,sig_gamma_2,sig_beta_2]

        if no_sig:
            sig_gamma_1, sig_gamma_2, sig_beta_1, sig_beta_2 = None, None, None, None


        x = F.relu(self.conv1(x, edge_index, weights['encoder.conv1.weight'],\
                weights['encoder.conv1.bias'], gamma=sig_gamma_1, beta=sig_beta_1)) # put gamma=sig_gamma_1, beta=sig_beta_1 if use sig
        if self.args.layer_norm:
            x = nn.LayerNorm(x.size()[1:], elementwise_affine=False)(x)
        if self.args.model in ['GAE']:
            x = self.conv2(x, edge_index,weights['encoder.conv2.weight'],\
                    weights['encoder.conv2.bias'],gamma=sig_gamma_2, beta=sig_beta_2) # put gamma=sig_gamma_2, beta=sig_beta_2 if use sig
            if self.args.layer_norm:
                x = nn.LayerNorm(x.size()[1:], elementwise_affine=False)(x)
            return x
        elif self.args.model in ['VGAE']:
            mu = self.conv_mu(x,edge_index,weights['encoder.conv_mu.weight'],\
                    weights['encoder.conv_mu.bias'], gamma=sig_gamma_2, beta=sig_beta_2)
            sig = self.conv_logvar(x,edge_index,weights['encoder.conv_logvar.weight'],\
                weights['encoder.conv_logvar.bias'], gamma=sig_gamma_2, beta=sig_beta_2)
            if self.args.layer_norm:
                mu = nn.LayerNorm(mu.size()[1:], elementwise_affine=False)(mu)
                sig = nn.LayerNorm(sig.size()[1:], elementwise_affine=False)(sig)
            return mu, sig

class MetaGatedSignatureEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(MetaGatedSignatureEncoder, self).__init__()
        self.args = args
        self.conv1 = MetaGatedGCNConv(in_channels, 2 * out_channels, gating=args.gating, cached=False)
        if args.model in ['GAE']:
            self.conv2 = MetaGatedGCNConv(2 * out_channels, out_channels, gating=args.gating, cached=False)
        elif args.model in ['VGAE']:
            self.conv_mu = MetaGatedGCNConv(2 * out_channels, out_channels, gating=args.gating, cached=False)
            self.conv_logvar = MetaGatedGCNConv(
                2 * out_channels, out_channels, gating=args.gating, cached=False)
        # in_channels is the input feature dim
        self.signature = GraphSignature(args, in_channels, out_channels)

    def forward(self, x, edge_index, weights, inner_loop=True):
        keys = list(weights.keys())
        sig_keys = [key for key in keys if 'signature' in key]
        if inner_loop:
            with torch.no_grad():
                sig_gamma_1, sig_beta_1, sig_gamma_2, sig_beta_2 = self.signature(x, edge_index, weights, sig_keys)
                self.cache_sig_out = [sig_gamma_1,sig_beta_1,sig_gamma_2,sig_beta_2,\
                                      torch.sigmoid(weights['encoder.conv1.gating_weights']),\
                                      torch.sigmoid(weights['encoder.conv_mu.gating_weights']),\
                                      torch.sigmoid(weights['encoder.conv_logvar.gating_weights'])]
        else:
            sig_gamma_1, sig_beta_1, sig_gamma_2, sig_beta_2 = self.signature(x, edge_index, weights, sig_keys)

        x = F.relu(self.conv1(x, edge_index,\
                weights['encoder.conv1.weight_1'],\
                weights['encoder.conv1.weight_2'],\
                weights['encoder.conv1.bias'],\
                weights['encoder.conv1.gating_weights'],\
                gamma=sig_gamma_1, beta=sig_beta_1))
        if self.args.layer_norm:
            x = nn.LayerNorm(x.size()[1:], elementwise_affine=False)(x)
        if self.args.model in ['GAE']:
            x =  self.conv2(x, edge_index,\
                    weights['encoder.conv_mu.weight_1'],\
                    weights['encoder.conv_mu.weight_2'],\
                    weights['encoder.conv_mu.bias'],\
                    weights['encoder.conv_mu.gating_weights'],\
                    gamma=sig_gamma_2, beta=sig_beta_2)
            if self.args.layer_norm:
                x = nn.LayerNorm(x.size()[1:], elementwise_affine=False)(x)
            return x
        elif self.args.model in ['VGAE']:
            mu = self.conv_mu(x,edge_index,\
                    weights['encoder.conv_mu.weight_1'],\
                    weights['encoder.conv_mu.weight_2'],\
                    weights['encoder.conv_mu.bias'],\
                    weights['encoder.conv_mu.gating_weights'],\
                    gamma=sig_gamma_2, beta=sig_beta_2)
            sig = self.conv_logvar(x,edge_index,\
                    weights['encoder.conv_logvar.weight_1'],\
                    weights['encoder.conv_logvar.weight_2'],\
                    weights['encoder.conv_logvar.bias'],\
                    weights['encoder.conv_logvar.gating_weights'],\
                    gamma=sig_gamma_2, beta=sig_beta_2)
            if self.args.layer_norm:
                mu = nn.LayerNorm(mu.size()[1:], elementwise_affine=False)(mu)
                sig = nn.LayerNorm(sig.size()[1:], elementwise_affine=False)(sig)
            return mu, sig

class MetaEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(MetaEncoder, self).__init__()
        self.args = args
        self.conv1 = MetaGCNConv(in_channels, 2 * out_channels, cached=False)
        if args.model in ['GAE']:
            self.conv2 = MetaGCNConv(2 * out_channels, out_channels, cached=False)
        elif args.model in ['VGAE']:
            self.conv_mu = MetaGCNConv(2 * out_channels, out_channels, cached=False)
            self.conv_logvar = MetaGCNConv(
                2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index, weights, inner_loop=True):
        x = F.relu(self.conv1(x, edge_index, \
                weights['encoder.conv1.weight'],weights['encoder.conv1.bias']))
        if self.args.model in ['GAE']:
            return self.conv2(x, edge_index,\
                    weights['encoder.conv2.weight'],weights['encoder.conv2.bias'])
        elif self.args.model in ['VGAE']:
            return self.conv_mu(x,edge_index,weights['encoder.conv_mu.weight'],\
                    weights['encoder.conv_mu.bias']),\
                self.conv_logvar(x,edge_index,weights['encoder.conv_logvar.weight'],\
                weights['encoder.conv_logvar.bias'])

class Net(torch.nn.Module):
    def __init__(self,train_dataset):
        super(Net, self).__init__()
        self.conv1 = GATConv(train_dataset.num_features, 256, heads=4)
        self.lin1 = torch.nn.Linear(train_dataset.num_features, 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(
            4 * 256, train_dataset.num_classes, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, train_dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x

