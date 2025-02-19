import torch, copy
import torch.nn.functional as F
import torch.nn as nn
# import multiprocessing as mp
import numpy as np
import itertools
import lib.utils as utils

class AVWGCN(nn.Module):
    def __init__(self, args, dim_in, dim_out, embed_dim):
        super(AVWGCN, self).__init__()
        self.args = args
        self.weights_pool = nn.Parameter(torch.FloatTensor(args.num_clients, embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(args.num_clients, embed_dim, dim_out))
        # self.norm1d = nn.ModuleList([nn.BatchNorm1d(args.rnn_units + args.input_dim) for _ in range(args.num_clients)])

    def forward(self, x, node_embeddings, poly_coefficients, mask):
        # node_embeddings: n d
        # x: b n c
        assert self.args.active_mode in ['sprtrelu', 'adptpolu']
        x = x * mask
        
        ret_per = [self.gen_upload(x[:,self.args.nodes_per[cid],:], node_embeddings[self.args.nodes_per[cid],:])
                        for cid in range(self.args.num_clients)]
        transformed_E_per, EH_per = [list(item) for item in zip(*ret_per)]

        if self.args.active_mode == "sprtrelu":
            ret_per = []
            for cid in range(self.args.num_clients):
                prob_tensor = torch.ones(1, self.args.num_clients).to(self.args.device) * (1 - self.args.inter_dropout)
                bernoulli_tensor = torch.bernoulli(prob_tensor)
                sum_EH = torch.einsum("kn,nbdc->kbdc", bernoulli_tensor, torch.stack(EH_per))[0]
                if bernoulli_tensor[0][cid].item() == 0.0:
                    sum_EH += EH_per[cid]
                else:
                    assert bernoulli_tensor[0][cid].item() == 1.0, bernoulli_tensor[0][cid].item()
                ret_per.append(self.recv_fwd(cid, node_embeddings[self.args.nodes_per[cid],:], x[:,self.args.nodes_per[cid],:], transformed_E_per[cid], sum_EH))

        elif self.args.active_mode == "adptpolu":
            ret_per = []
            for cid in range(self.args.num_clients):
                prob_tensor = torch.ones(1, self.args.num_clients).to(self.args.device) * (1 - self.args.inter_dropout)
                bernoulli_tensor = torch.bernoulli(prob_tensor)
                sum_EH = [torch.einsum("kn,nbdc->kbdc", bernoulli_tensor, torch.stack(item))[0] for item in zip(*EH_per)]
                if bernoulli_tensor[0][cid].item() == 0.0:
                    sum_EH = [sum_EH[i]+EH_per[cid][i] for i in range(len(sum_EH))]
                else:
                    assert bernoulli_tensor[0][cid].item() == 1.0, bernoulli_tensor[0][cid].item()
                ret_per.append(self.recv_fwd(cid, node_embeddings[self.args.nodes_per[cid],:], x[:,self.args.nodes_per[cid],:], transformed_E_per[cid], sum_EH, poly_coefficients[cid]))

        return torch.concatenate(ret_per, dim=1)

    def gen_upload(self, x, node_embeddings, mask):
        E = node_embeddings # n d
        H = x # b n c
        if self.args.active_mode == "sprtrelu":
            transformed_E = torch.relu(E)
            EH = torch.einsum("dn,bnc->bdc", transformed_E.transpose(0,1), H)
        elif self.args.active_mode == "adptpolu":
            transformed_E = [self.transform(k, E) for k in range(self.args.act_k+1)]
            EH = [torch.einsum("dn,bnc->bdc", e.transpose(0,1), H) for e in transformed_E]
        return transformed_E, EH

    def recv_fwd(self, cid, E, H, transformed_E, sum_EH, P=None):

        if self.args.active_mode == "sprtrelu":
            Z = H + torch.einsum("nd,bdc->bnc", transformed_E, sum_EH)
        elif self.args.active_mode == "adptpolu":
            Z = torch.stack([torch.einsum("nd,bdc->bnc", transformed_E[i], sum_EH[i]) for i in range(self.args.act_k+1)])
            Z = torch.einsum('ak,kbnc->abnc', P, Z)[0]
            Z = H + Z

        # print(Z.shape)
        # Z = self.norm1d[cid](Z.transpose(1,2)).transpose(1,2)
        
        weights = torch.einsum('nd,dio->nio', E, self.weights_pool[cid])  #N, dim_in, dim_out
        bias = torch.matmul(E, self.bias_pool[cid])                       #N, dim_out
        x_gconv = torch.einsum('bni,nio->bno', Z, weights) + bias     #b, N, dim_out
        return x_gconv

    def cartesian_prod(self, A, B):
        transformed = torch.stack(list(map(torch.cartesian_prod, A, B)))
        transformed = transformed[...,0] * transformed[...,1]
        return transformed

    def transform(self, k, E):
        ori_k = k
        transformed = torch.ones(E.shape[0], 1).to(E.device)
        cur_pow = self.cartesian_prod(transformed, E)
        while k > 0:
            if k % 2 == 1:
                transformed = self.cartesian_prod(transformed, cur_pow)
            cur_pow = self.cartesian_prod(cur_pow, cur_pow)
            k //= 2
        assert transformed.shape[0] == E.shape[0], (transformed.shape[0], E.shape[0])
        assert transformed.shape[1] == E.shape[1]**ori_k, (transformed.shape[1], E.shape[1], ori_k)
        return transformed

    def fedavg(self):
        mean_w = torch.sum(self.weights_pool, dim=0, keepdim=True).data / self.args.num_clients
        mean_b = torch.sum(self.bias_pool, dim=0, keepdim=True).data / self.args.num_clients

        self.weights_pool = nn.Parameter(torch.repeat_interleave(mean_w, self.args.num_clients, dim=0), requires_grad=True).to(mean_w.device)
        self.bias_pool = nn.Parameter(torch.repeat_interleave(mean_b, self.args.num_clients, dim=0), requires_grad=True).to(mean_b.device)

        # mean_norm = utils.avg([norm.state_dict() for norm in self.norm1d])
        # for i in range(self.args.num_clients): self.norm1d[i].load_state_dict(copy.deepcopy(mean_norm))
