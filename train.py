import utils
from parse_args import args
from task import predict_crime, clustering, predict_check
from model import EUPAC
import random
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F


seed = 2022
torch.manual_seed(seed=seed)
np.random.seed(seed)
random.seed(seed)


def add_noise_replace(data, replace_ratio):
    noisy_data = np.copy(data)
    num_replace = int(replace_ratio * len(data))
    replace_indices = np.random.choice(len(data), size=num_replace, replace=False)
    replacement_values = np.random.rand(num_replace)
    noisy_data[replace_indices] = replacement_values

    return noisy_data


replace_ratio = 0.2
poi_similarity, s_adj, d_adj, mobility, neighbor = utils.load_data()
poi_edge_index = utils.create_graph(poi_similarity, args.importance_k)
s_edge_index = utils.create_graph(s_adj, args.importance_k)
d_edge_index = utils.create_graph(d_adj, args.importance_k)
n_edge_index = utils.create_neighbor_graph(neighbor)
poi_edge_index = torch.tensor(poi_edge_index, dtype=torch.long).to(args.device)
s_edge_index = torch.tensor(s_edge_index, dtype=torch.long).to(args.device)
d_edge_index = torch.tensor(d_edge_index, dtype=torch.long).to(args.device)
n_edge_index = torch.tensor(n_edge_index, dtype=torch.long).to(args.device)

mobility = torch.tensor(mobility, dtype=torch.float32).to(args.device)
poi_similarity = torch.tensor(
    poi_similarity, dtype=torch.float32).to(args.device)

features = torch.randn(args.regions_num, args.embedding_size).to(args.device)
poi_r = torch.randn(args.embedding_size).to(args.device)
s_r = torch.randn(args.embedding_size).to(args.device)
d_r = torch.randn(args.embedding_size).to(args.device)
n_r = torch.randn(args.embedding_size).to(args.device)
rel_emb = [poi_r, s_r, d_r, n_r]
edge_index = [poi_edge_index, s_edge_index, d_edge_index, n_edge_index]


def pairwise_inner_product(mat_1, mat_2):
    n, m = mat_1.shape
    mat_expand = torch.unsqueeze(mat_2, 0)
    mat_expand = mat_expand.expand(n, n, m)
    mat_expand = mat_expand.permute(1, 0, 2)
    inner_prod = torch.mul(mat_expand, mat_1)
    inner_prod = torch.sum(inner_prod, axis=-1)
    return inner_prod


def mob_loss(s_emb, d_emb, mob):
    inner_prod = torch.mm(s_emb, d_emb.T)
    ps_hat = F.softmax(inner_prod, dim=-1)
    inner_prod = torch.mm(d_emb, s_emb.T)
    pd_hat = F.softmax(inner_prod, dim=-1)
    loss = torch.sum(-torch.mul(mob, torch.log(ps_hat)) -
                     torch.mul(mob, torch.log(pd_hat)))
    return loss

def train(net):
    optimizer = optim.Adam(
        net.parameters(), lr=args.learning_rate, weight_decay=5e-3)
    loss_fn1 = torch.nn.TripletMarginLoss()
    loss_fn2 = torch.nn.MSELoss()
    self_weight = 0.15
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        region_emb, n_emb, poi_emb, s_emb, d_emb, cont_loss = net(
            features, rel_emb, edge_index, mobility, poi_similarity, neighbor)
        pos_idx, neg_idx = utils.pair_sample(neighbor)
        geo_loss = loss_fn1(n_emb, n_emb[pos_idx], n_emb[neg_idx])
        m_loss = mob_loss(s_emb, d_emb, mobility)
        poi_loss = loss_fn2(torch.mm(poi_emb, poi_emb.T), poi_similarity)
        loss = poi_loss + m_loss + geo_loss
        loss = (1 - self_weight) * cont_loss + loss * self_weight
        loss.backward()
        optimizer.step()
