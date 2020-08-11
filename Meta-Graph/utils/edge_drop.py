import torch
import torch.nn as nn
import numpy as np
import os, sys
from enum import Enum
import matplotlib.pyplot as plt
import pydot

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def plot_graph(d, name):
    d = d.data.cpu().numpy()
    graph = pydot.Dot(graph_type='graph')
    s = set()
    for index in range(d.shape[1]):
        if not (d[0][index], d[1][index]) in s or not (d[1][index], d[0][index]) in s:
            node = pydot.Node(str(d[0][index]), style="filled", fillcolor="green")
            graph.add_node(node)
            edge = pydot.Edge(node, str(d[1][index]))
            graph.add_edge(edge)
            s.add((d[0][index], d[1][index]))
            s.add((d[1][index], d[0][index]))
    graph.write_png(name)


class DropMode(Enum):
    EQUAL = 'equal'
    WEIGHTED = 'weighted'


class EdgeDrop(nn.Module):
    def __init__(self, keep_prob, mode=DropMode.EQUAL, plot=False):
        super(EdgeDrop, self).__init__()
        self.keep_prob = keep_prob
        self.mode = mode
        self.plot = plot

    def forward(self, x):
        if self.mode == DropMode.EQUAL:
            alpha = torch.ones_like(x[0]) * self.keep_prob
            edge_index = torch.bernoulli(alpha)
            indices = torch.where(edge_index == 1)[0]
            x = x[:, indices]
            print(x.shape)
            return x
        elif self.mode == DropMode.WEIGHTED:
            counts, drop_rate, unique, upper_line, upper_quartile = self.get_connection_detail(x)

            upper_indices = np.where(counts >= upper_line)[0]
            lower_indices = np.where(counts < upper_line)[0]
            upper_mean = np.mean(counts[upper_indices], axis=0)
            upper_nodes = unique[upper_indices]
            lower_nodes = unique[lower_indices]

            rate_adj = [(upper_mean / upper_quartile) * drop_rate] * len(upper_nodes)
            drop_rate = [1. - self.keep_prob] * len(lower_nodes)

            upper_dict = dict(zip(upper_nodes, rate_adj))
            lower_dict = dict(zip(lower_nodes, drop_rate))
            upper_dict.update(lower_dict)
            dictionary = upper_dict
            nodes = x.detach().cpu().numpy()
            alpha = list(map(lambda i: dictionary[i], nodes[0]))
            alpha = torch.tensor(np.array(alpha, dtype=float))
            edge_index = torch.bernoulli(alpha)
            indices = torch.where(edge_index == 0)[0]
            x = x[:, indices]
            if self.plot:
                self.get_connection_detail(x)
            return x

    def get_connection_detail(self, x):
        (unique, counts) = np.unique(x[0], return_counts=True)
        drop_rate = 1. - self.keep_prob
        Q3 = np.percentile(counts, 75, axis=0)
        Q1 = np.percentile(counts, 25, axis=0)
        IQR = Q3 - Q1
        upper_quartile = Q3 + 1.5 * IQR
        upper_line = [upper_quartile] * len(counts)
        if self.plot:
            plt.plot(counts, c='blue')
            plt.plot(upper_line, c='red')
            plt.show()
        return counts, drop_rate, unique, upper_line, upper_quartile


if __name__ == '__main__':
    edge_drop = EdgeDrop(keep_prob=.7, mode=DropMode.WEIGHTED, plot=True)
    with open('data.npy', 'rb') as f:
        x = np.load(f)
        x = torch.tensor(x)
        plot_graph(x, name='before_graph_weighted.png')
        x = edge_drop(x)
        plot_graph(x, name='after_graph_weighted.png')

