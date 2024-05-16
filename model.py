from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.init as init
import utils
import numpy as np
import torch
import torch.nn.functional as F


def pairwise_inner_product(mat_1, mat_2):
    n, m = mat_1.shape  # (180, 144)
    mat_expand = torch.unsqueeze(mat_2, 0)
    mat_expand = mat_expand.expand(n, n, m)
    mat_expand = mat_expand.permute(1, 0, 2)
    inner_prod = torch.mul(mat_expand, mat_1)
    inner_prod = torch.sum(inner_prod, axis=-1)
    return inner_prod


class RelationGCN(nn.Module):
    def __init__(self, embedding_size, dropout, gcn_layers):
        super(RelationGCN, self).__init__()
        self.gcn_layers = gcn_layers
        self.dropout = dropout

        self.gcns = nn.ModuleList([GCNConv(in_channels=embedding_size, out_channels=embedding_size)
                                   for _ in range(self.gcn_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(embedding_size)
                                  for _ in range(self.gcn_layers - 1)])
        self.relation_transformation = nn.ModuleList([nn.Linear(embedding_size, embedding_size)
                                                      for _ in range(self.gcn_layers)])

    def forward(self, features, rel_emb, edge_index, is_training=True):
        n_emb = features
        poi_emb = features
        s_emb = features
        d_emb = features
        poi_r, s_r, d_r, n_r = rel_emb
        poi_edge_index, s_edge_index, d_edge_index, n_edge_index = edge_index
        for i in range(self.gcn_layers - 1):
            tmp = n_emb
            n_emb = tmp + F.leaky_relu(self.bns[i](
                self.gcns[i](torch.multiply(n_emb, n_r), n_edge_index)))
            n_r = self.relation_transformation[i](n_r)
            if is_training:
                n_emb = F.dropout(n_emb, p=self.dropout)

            tmp = poi_emb
            poi_emb = tmp + F.leaky_relu(self.bns[i](
                self.gcns[i](torch.multiply(poi_emb, poi_r), poi_edge_index)))
            poi_r = self.relation_transformation[i](poi_r)
            if is_training:
                poi_emb = F.dropout(poi_emb, p=self.dropout)

            tmp = s_emb
            s_emb = tmp + F.leaky_relu(self.bns[i](
                self.gcns[i](torch.multiply(s_emb, s_r), s_edge_index)))
            s_r = self.relation_transformation[i](s_r)
            if is_training:
                s_emb = F.dropout(s_emb, p=self.dropout)

            tmp = d_emb
            d_emb = tmp + F.leaky_relu(self.bns[i](
                self.gcns[i](torch.multiply(d_emb, d_r), d_edge_index)))
            d_r = self.relation_transformation[i](d_r)
            if is_training:
                d_emb = F.dropout(d_emb, p=self.dropout)

        n_emb = self.gcns[-1](torch.multiply(n_emb, n_r), n_edge_index)
        poi_emb = self.gcns[-1](torch.multiply(poi_emb, poi_r), poi_edge_index)
        s_emb = self.gcns[-1](torch.multiply(s_emb, s_r), s_edge_index)
        d_emb = self.gcns[-1](torch.multiply(d_emb, d_r), d_edge_index)
        ##################################################
        n_r = self.relation_transformation[-1](n_r)
        poi_r = self.relation_transformation[-1](poi_r)
        s_r = self.relation_transformation[-1](s_r)
        d_r = self.relation_transformation[-1](d_r)

        return n_emb, poi_emb, s_emb, d_emb, n_r, poi_r, s_r, d_r


class FusionView(nn.Module):
    def __init__(self, d_model, S, num_views):
        super(FusionView, self).__init__()
        self.num_views = num_views

    def forward(self, views):
        outs = []
        for i in range(self.num_views):
            output = self.external_attentions[i](views[i])
            outs.append(output)
        output = torch.mean(torch.stack(outs, dim=0), dim=0)
        return output


class CrossLayer(nn.Module):
    def __init__(self, regions_num, embedding_size):
        super(CrossLayer, self).__init__()
        self.alpha_n = nn.Parameter(torch.tensor(0.8))
        self.alpha_poi = nn.Parameter(torch.tensor(0.8))
        self.alpha_d = nn.Parameter(torch.tensor(0.8))
        self.alpha_s = nn.Parameter(torch.tensor(0.8))
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_size, num_heads=4)

    def forward(self, n_emb, poi_emb, s_emb, d_emb):
        stk_emb = torch.stack((n_emb, poi_emb, d_emb, s_emb))
        fusion, _ = self.attn(stk_emb, stk_emb, stk_emb)
        n_f = fusion[0] * self.alpha_n + (1 - self.alpha_n) * n_emb
        poi_f = fusion[1] * self.alpha_poi + (1 - self.alpha_poi) * poi_emb
        d_f = fusion[2] * self.alpha_d + (1 - self.alpha_d) * d_emb
        s_f = fusion[3] * self.alpha_s + (1 - self.alpha_s) * s_emb
        return n_f, poi_f, s_f, d_f


class AttentionFusionLayer(nn.Module):
    def __init__(self, embedding_size):
        super(AttentionFusionLayer, self).__init__()
        self.q = nn.Parameter(torch.randn(embedding_size))
        self.fusion_lin = nn.Linear(embedding_size, embedding_size)

    def forward(self, n_f, poi_f, s_f, d_f):
        n_w = torch.mean(torch.sum(F.leaky_relu(
            self.fusion_lin(n_f)) * self.q, dim=1))
        poi_w = torch.mean(torch.sum(F.leaky_relu(
            self.fusion_lin(poi_f)) * self.q, dim=1))
        s_w = torch.mean(torch.sum(F.leaky_relu(
            self.fusion_lin(s_f)) * self.q, dim=1))
        d_w = torch.mean(torch.sum(F.leaky_relu(
            self.fusion_lin(d_f)) * self.q, dim=1))

        w_stk = torch.stack((n_w, poi_w, s_w, d_w))
        w = torch.softmax(w_stk, dim=0)

        region_feature = w[0] * n_f + w[1] * poi_f + w[2] * s_f + w[3] * d_f
        return region_feature


class EUPAC(nn.Module):
    def __init__(self, regions_num, embedding_size, dropout, gcn_layers):
        super(EUPAC, self).__init__()
        self.decoder = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(0.5)
        self.neg_eps = 0.5
        self.gamma = 0.5
        self.tau = 4
        self.pos_eps = 0.5
        self.dense = nn.Linear(in_features=embedding_size, out_features=embedding_size)
        self.relation_gcns = RelationGCN(embedding_size, dropout, gcn_layers)
        self.projection = nn.Sequential(nn.Linear(embedding_size, embedding_size),
                                        nn.ReLU())
        self.cross_layer = CrossLayer(regions_num, embedding_size)

        self.fusion_layer = AttentionFusionLayer(embedding_size)

    def generate_adv_mob(self, Anchor_hiddens, lm_labels, mob_label):
        Anchor_hiddens = Anchor_hiddens.detach()
        lm_labels = lm_labels.detach()
        Anchor_hiddens.requires_grad_(True)

        Anchor_logits = self.dense(Anchor_hiddens)

        Anchor_logits = F.log_softmax(Anchor_logits, -1)
        inner_prod = pairwise_inner_product(Anchor_logits, lm_labels)
        softmax1 = nn.Softmax(dim=-1)
        phat = softmax1(inner_prod)
        loss_adv = torch.sum(-torch.mul(mob_label, torch.log(phat + 0.0001))).requires_grad_()
        loss_adv.backward()  # 计算了 loss_adv 相对于 Anchor的梯度。
        dec_grad = Anchor_hiddens.grad.detach()
        # 计算了 dec_grad 的 L2 范数，以获得每个隐藏状态的模
        l2_norm = torch.norm(dec_grad, dim=-1)

        # 梯度标准化。将梯度 dec_grad 中的每个元素除以对应位置的 L2 范数,以确保梯度的模为1，并避免在除法运算中出现除以零
        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)
        # 负扰动系数
        perturbed_Anc = Anchor_hiddens + self.neg_eps * dec_grad.detach()
        perturbed_Anc = perturbed_Anc  # [b,t,d]

        self.zero_grad()
        return perturbed_Anc

    def generate_adv_poi(self, Anchor_hiddens, lm_labels):
        Anchor_hiddens = Anchor_hiddens.detach()  # 不会再跟踪梯度信
        lm_labels = lm_labels.detach()
        Anchor_hiddens.requires_grad_(True)  # 重新启用梯度跟踪，以便进行梯度计算

        Anchor_logits = self.dense(Anchor_hiddens)

        Anchor_logits = F.log_softmax(Anchor_logits, -1)
        inner_prod = pairwise_inner_product(Anchor_logits, Anchor_logits)
        loss_adv = F.mse_loss(inner_prod, lm_labels).requires_grad_()
        loss_adv.backward()  # 计算了 loss_adv 相对于 Anchor的梯度。
        dec_grad = Anchor_hiddens.grad.detach()
        # 计算了 dec_grad 的 L2 范数，以获得每个隐藏状态的模
        l2_norm = torch.norm(dec_grad, dim=-1)

        # 梯度标准化。将梯度 dec_grad 中的每个元素除以对应位置的 L2 范数,以确保梯度的模为1，并避免在除法运算中出现除以零
        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)
        # 负扰动系数
        perturbed_Anc = Anchor_hiddens + self.neg_eps * dec_grad.detach()
        perturbed_Anc = perturbed_Anc  # [b,t,d]

        self.zero_grad()
        return perturbed_Anc

    def generate_adv_nei(self, Anchor_hiddens, lm_labels):
        Anchor_hiddens = Anchor_hiddens.detach()  # 不会再跟踪梯度信
        # lm_labels = lm_labels.detach()
        Anchor_hiddens.requires_grad_(True)  # 重新启用梯度跟踪，以便进行梯度计算
        pos_idx, neg_idx = utils.pair_sample(lm_labels)
        loss_fn1 = torch.nn.TripletMarginLoss()
        Anchor_logits = self.dense(Anchor_hiddens)
        Anchor_logits = F.log_softmax(Anchor_logits, -1)
        loss_adv = loss_fn1(Anchor_logits, Anchor_logits[pos_idx], Anchor_logits[neg_idx]).requires_grad_()
        loss_adv.backward()  # 计算了 loss_adv 相对于 Anchor的梯度。
        dec_grad = Anchor_hiddens.grad.detach()
        # 计算了 dec_grad 的 L2 范数，以获得每个隐藏状态的模
        l2_norm = torch.norm(dec_grad, dim=-1)

        # 梯度标准化。将梯度 dec_grad 中的每个元素除以对应位置的 L2 范数,以确保梯度的模为1，并避免在除法运算中出现除以零
        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)
        # 负扰动系数
        perturbed_Anc = Anchor_hiddens + self.neg_eps * dec_grad.detach()
        perturbed_Anc = perturbed_Anc  # [b,t,d]

        self.zero_grad()
        return perturbed_Anc

    def generate_cont_adv(self, STNPos_hiddens,
                          Anchor_hiddens, pred,
                          tau, eps):
        # 张量从计算图中分离，使其成为不可微的常量
        STNPos_hiddens = STNPos_hiddens.detach()
        Anchor_hiddens = Anchor_hiddens.detach()
        Anchor_logits = pred.detach()
        # 为需要计算梯度，以便后续计算梯度
        STNPos_hiddens.requires_grad = True
        Anchor_logits.requires_grad = True  # 模型对锚点的分类置信度(未归一化分数)
        Anchor_hiddens.requires_grad = True

        avg_STNPos = self.projection(STNPos_hiddens)
        avg_Anchor = self.projection(Anchor_hiddens)

        cos = nn.CosineSimilarity(dim=-1)

        # 最小化这两个表示之间的余弦相似度
        logits = cos(avg_STNPos.unsqueeze(1), avg_Anchor.unsqueeze(0)) / tau

        cont_crit = nn.CrossEntropyLoss()
        labels = torch.arange(avg_STNPos.size(0),
                              device=STNPos_hiddens.device)
        loss_cont_adv = cont_crit(logits, labels)
        loss_cont_adv.backward()

        dec_grad = Anchor_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)
        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)
        # 正扰动系数
        perturb_Anchor_hidden = Anchor_hiddens + eps * dec_grad
        perturb_Anchor_hidden = perturb_Anchor_hidden.detach()
        perturb_Anchor_hidden.requires_grad = True
        perturb_logits = self.dense(perturb_Anchor_hidden)
        # perturb_logits = nn.LogSoftmax(dim=1)(perturb_logits)

        true_probs = F.softmax(Anchor_logits, -1)
        # true_probs = true_probs * dec_mask.unsqueeze(-1).float()

        perturb_log_probs = F.log_softmax(perturb_logits, -1)

        kl_crit = nn.KLDivLoss(reduction="sum")
        vocab_size = Anchor_logits.size(-1)
        # 比较两个概率分布之间的差异
        kl = kl_crit(perturb_log_probs.view(-1, vocab_size),
                     true_probs.view(-1, vocab_size))
        kl = kl / torch.tensor(true_probs.shape[0]).float()
        kl.backward()

        kl_grad = perturb_Anchor_hidden.grad.detach()

        l2_norm = torch.norm(kl_grad, dim=-1)

        kl_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_Anchor_hidden = perturb_Anchor_hidden - eps * kl_grad

        return perturb_Anchor_hidden

    # def replace_embedding(self, original_embedding, dataset_embeddings):
    #     replacement_idx = np.random.choice(len(dataset_embeddings))
    #     replacement_embedding = dataset_embeddings[replacement_idx]
    #     new_embedding = original_embedding.clone()
    #     new_embedding[:] = replacement_embedding
    #     return new_embedding

    def forward(self, features, rel_emb, edge_index, mobility, poi_similarity, neighbor, is_training=True):
        poi_emb, s_emb, d_emb, n_emb, poi_r, s_r, d_r, n_r = self.relation_gcns(features, rel_emb, edge_index,
                                                                                is_training)
        mob_noise = torch.normal(0, 0.01, s_emb.shape).to(s_emb.device)
        poi_noise = torch.normal(0, 0.01, poi_emb.shape).to(poi_emb.device)
        nei_noise = torch.normal(0, 0.01, n_emb.shape).to(n_emb.device)
        s_emb_STNPos = s_emb + mob_noise
        d_emb_STNPos = d_emb + mob_noise
        poi_emb_STNPos = poi_emb + poi_noise
        nei_emb_STNPos = n_emb + nei_noise
        n_f, poi_f, s_f, d_f = self.cross_layer(n_emb, poi_emb, s_emb, d_emb)

        n_f_STNPos, poi_f_STNPos, s_f_STNPos, d_f_STNPos = self.cross_layer(nei_emb_STNPos, poi_emb_STNPos,
                                                                            s_emb_STNPos, d_emb_STNPos)
        n_f_STNPos = self.decoder(n_f_STNPos)
        poi_f_STNPos = self.decoder(poi_f_STNPos)
        s_f_STNPos = self.decoder(s_f_STNPos)
        d_f_STNPos = self.decoder(d_f_STNPos)
        dense_s = self.dense(s_f)
        dense_d = self.dense(d_f)
        dense_poi = self.dense(poi_f)
        dense_nei = self.dense(n_f)
        s_out = self.dropout(s_f)
        d_out = self.dropout(d_f)
        poi_out = self.dropout(poi_f)
        nei_out = self.dropout(n_f)
        s_out_STNPos = self.dropout(s_f_STNPos)
        d_out_STNPos = self.dropout(d_f_STNPos)
        poi_out_STNPos = self.dropout(poi_f_STNPos)
        nei_out_STNPos = self.dropout(n_f_STNPos)
        avg_s_out_STNPos = self.projection(s_out_STNPos)
        avg_d_out_STNPos = self.projection(d_out_STNPos)
        avg_poi_out_STNPos = self.projection(poi_out_STNPos)
        avg_nei_out_STNPos = self.projection(nei_out_STNPos)
        avg_s_out = self.projection(s_out)
        avg_d_out = self.projection(d_out)
        avg_poi_out = self.projection(poi_out)
        avg_nei_out = self.projection(nei_out)
        cos = nn.CosineSimilarity(dim=-1)
        cont_crit = nn.CrossEntropyLoss()

        adv_tri_s = self.generate_adv_mob(avg_s_out, avg_d_out, mobility)
        adv_tri_d = self.generate_adv_mob(avg_d_out, avg_s_out, mobility)
        adv_tri_poi = self.generate_adv_poi(avg_poi_out, poi_similarity)
        adv_tri_nei = self.generate_adv_nei(avg_nei_out, neighbor)
        avg_adv_tri_s = self.projection(adv_tri_s)
        avg_adv_tri_d = self.projection(adv_tri_d)
        avg_adv_tri_poi = self.projection(adv_tri_poi)
        avg_adv_tri_nei = self.projection(adv_tri_nei)
        adv_sim_s = cos(avg_s_out_STNPos, avg_adv_tri_s).unsqueeze(1)
        adv_sim_d = cos(avg_d_out_STNPos, avg_adv_tri_d).unsqueeze(1)
        adv_sim_poi = cos(avg_poi_out_STNPos, avg_adv_tri_poi).unsqueeze(1)
        adv_sim_nei = cos(avg_nei_out_STNPos, avg_adv_tri_nei).unsqueeze(1)

        adv_devcopy_s = self.generate_cont_adv(s_out_STNPos,
                                                 s_out, dense_s,
                                                 self.tau, self.pos_eps)
        adv_devcopy_d = self.generate_cont_adv(d_out_STNPos,
                                                 d_out, dense_d,
                                                 self.tau, self.pos_eps)
        adv_devcopy_poi = self.generate_cont_adv(poi_out_STNPos,
                                                   poi_out, dense_poi,
                                                   self.tau, self.pos_eps)
        adv_devcopy_nei = self.generate_cont_adv(nei_out_STNPos,
                                                   nei_out, dense_nei,
                                                   self.tau, self.pos_eps)
        avg_adv_devcopy_s = self.projection(adv_devcopy_s)
        avg_adv_devcopy_d = self.projection(adv_devcopy_d)
        avg_adv_devcopy_poi = self.projection(adv_devcopy_poi)
        avg_adv_devcopy_nei = self.projection(adv_devcopy_nei)
        pos_sim_s = cos(avg_s_out_STNPos, avg_adv_devcopy_s).unsqueeze(-1)
        pos_sim_t = cos(avg_d_out_STNPos, avg_adv_devcopy_d).unsqueeze(-1)
        pos_sim_poi = cos(avg_poi_out_STNPos, avg_adv_devcopy_poi).unsqueeze(-1)
        pos_sim_nei = cos(avg_nei_out_STNPos, avg_adv_devcopy_nei).unsqueeze(-1)

        sim_matrix_s = cos(avg_s_out_STNPos.unsqueeze(1),  # torch.Size([180, 180])
                           avg_s_out.unsqueeze(0))
        sim_matrix_d = cos(avg_d_out_STNPos.unsqueeze(1),
                           avg_d_out.unsqueeze(0))
        sim_matrix_poi = cos(avg_poi_out_STNPos.unsqueeze(1),
                             avg_poi_out.unsqueeze(0))
        sim_matrix_nei = cos(avg_nei_out_STNPos.unsqueeze(1),
                             avg_nei_out.unsqueeze(0))

        logits_s = torch.cat([sim_matrix_s, adv_sim_s], 1) / self.tau
        logits_d = torch.cat([sim_matrix_d, adv_sim_d], 1) / self.tau
        logits_poi = torch.cat([sim_matrix_poi, adv_sim_poi], 1) / self.tau
        logits_nei = torch.cat([sim_matrix_nei, adv_sim_nei], 1) / self.tau
        identity = torch.eye(180, device=s_out.device)

        pos_sim_s = identity * pos_sim_s
        pos_sim_t = identity * pos_sim_t
        pos_sim_poi = identity * pos_sim_poi
        pos_sim_nei = identity * pos_sim_nei

        neg_sim_s = sim_matrix_s.masked_fill(identity == 1, 0)
        neg_sim_t = sim_matrix_d.masked_fill(identity == 1, 0)
        neg_sim_poi = sim_matrix_poi.masked_fill(identity == 1, 0)
        neg_sim_chk = sim_matrix_nei.masked_fill(identity == 1, 0)

        new_sim_matrix_s = pos_sim_s + neg_sim_s
        new_sim_matrix_t = pos_sim_t + neg_sim_t
        new_sim_matrix_poi = pos_sim_poi + neg_sim_poi
        new_sim_matrix_chk = pos_sim_nei + neg_sim_chk

        new_logits_s = torch.cat([new_sim_matrix_s, adv_sim_s], 1)
        new_logits_d = torch.cat([new_sim_matrix_t, adv_sim_d], 1)
        new_logits_poi = torch.cat([new_sim_matrix_poi, adv_sim_poi], 1)
        new_logits_nei = torch.cat([new_sim_matrix_chk, adv_sim_nei], 1)

        labels = torch.arange(180, device=s_out.device)

        cont_loss_s = cont_crit(logits_s, labels)
        cont_loss_d = cont_crit(logits_d, labels)
        cont_loss_poi = cont_crit(logits_poi, labels)
        cont_loss_nei = cont_crit(logits_nei, labels)

        new_cont_loss_s = cont_crit(new_logits_s, labels)
        new_cont_loss_t = cont_crit(new_logits_d, labels)
        new_cont_loss_poi = cont_crit(new_logits_poi, labels)
        new_cont_loss_chk = cont_crit(new_logits_nei, labels)

        cont_loss = self.gamma * (cont_loss_s + cont_loss_d + cont_loss_poi + cont_loss_nei) + (1-self.gamma) * (
                new_cont_loss_s + new_cont_loss_t + new_cont_loss_poi + new_cont_loss_chk)
        region_feature = self.fusion_layer(n_f, poi_f, s_f, d_f)

        n_f = torch.multiply(region_feature, n_r)
        poi_f = torch.multiply(region_feature, poi_r)
        s_f = torch.multiply(region_feature, s_r)
        d_f = torch.multiply(region_feature, d_r)

        return region_feature, n_f, poi_f, s_f, d_f, cont_loss
