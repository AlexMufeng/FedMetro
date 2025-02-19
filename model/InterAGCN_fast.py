import torch, copy
import torch.nn.functional as F
import torch.nn as nn
# import multiprocessing as mp
import numpy as np
import itertools
import lib.utils as utils

cnt = 0

class AVWGCN(nn.Module):
    def __init__(self, args, dim_in, dim_out, embed_dim):
        super(AVWGCN, self).__init__()
        self.args = args
        self.weights_pool = nn.Parameter(torch.FloatTensor(args.num_clients, embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(args.num_clients, embed_dim, dim_out))
        self.indices = [idx for sublist in args.nodes_per for idx in sublist]
        # self.norm1d = nn.ModuleList([nn.BatchNorm1d(args.rnn_units + args.input_dim) for _ in range(args.num_clients)])

    def forward(self, x, node_embeddings, poly_coefficients, mask):
        # node_embeddings: n d
        # x: b n c
        # mask: b n d
        assert self.args.active_mode in ['sprtrelu', 'adptpolu']
        assert self.args.inter_dropout in [0.0, 1.0]

        # if self.args.inter_dropout == 0.0:
        #     transformed_E, sum_EH = self.gen_upload(node_embeddings, x)
        # else:
        #     ret_per = [self.gen_upload(x[:,self.args.nodes_per[cid],:], node_embeddings[self.args.nodes_per[cid],:])
        #                 for cid in range(self.args.num_clients)]
        #     transformed_E_per, EH_per = [list(item) for item in zip(*ret_per)]

        #     prob_tensor = torch.ones(1, self.args.num_clients).to(self.args.device) * (1-self.args.inter_dropout)
        #     bernoulli_tensor = torch.bernoulli(prob_tensor)
        #     if self.args.active_mode == "sprtrelu":
        #         transformed_E = torch.concat(transformed_E_per, dim=0)
        #         sum_EH = torch.einsum("kn,nbdc->kbdc", bernoulli_tensor, torch.stack(EH_per))[0]
        #     elif self.args.active_mode == "adptpolu":
        #         transformed_E = [torch.concat(item, dim=0) for item in zip(*transformed_E_per)]
        #         sum_EH = [torch.einsum("kn,nbdc->kbdc", bernoulli_tensor, torch.stack(item))[0] for item in zip(*EH_per)]
        x = x * mask
        transformed_E, sum_EH, sp, tran = self.gen_upload(node_embeddings, x)

        if self.args.active_mode == "sprtrelu":
            ret_per = self.fast_recv_fwd(node_embeddings, x, transformed_E, sum_EH)
        elif self.args.active_mode == "adptpolu":
            ret_per = self.fast_recv_fwd(node_embeddings, x, transformed_E, sum_EH, poly_coefficients)
        ret_concat = torch.concatenate(ret_per, dim=1)
        ret = torch.zeros_like(ret_concat)
        ret[:, self.indices, :] = ret_concat
        return ret, sp, tran
    
    def gen_upload(self, E, H):
        # E: n d
        # H: b n c
        if self.args.active_mode == "sprtrelu":
            transformed_E = torch.relu(E)
            EH = torch.einsum("dn,bnc->bdc", transformed_E.transpose(0,1), H)
        elif self.args.active_mode == "adptpolu":
            transformed_E = [self.transform(k, E) for k in range(self.args.act_k+1)]
            EH = [torch.einsum("dn,bnc->bdc", e.transpose(0,1), H) for e in transformed_E]
            numerator, denominator = 0, 0
            for cid in range(self.args.num_clients):
                sum_EH = [torch.einsum("dn,bnc->bdc", e.transpose(0,1)[:, self.args.nodes_per[cid]], H[:, self.args.nodes_per[cid], :]) for e in transformed_E]
                for eh in sum_EH:
                    numerator += eh.nonzero().size(0)
                    if eh.dim() == 2:
                        denominator += eh.size(0)*eh.size(1)
                    else:
                        denominator += eh.size(0)*eh.size(1)*eh.size(2)
            sp = 1 - numerator/(denominator + 1e-10)
            numerator, denominator = 0, 0
            for eh in EH:
                numerator += eh.nonzero().size(0)
                if eh.dim() == 2:
                    denominator += eh.size(0)*eh.size(1)
                else:
                    denominator += eh.size(0)*eh.size(1)*eh.size(2)
            tran = (1 - sp + numerator/(denominator + 1e-10))/2
        return transformed_E, EH, sp, tran

    def fast_recv_fwd(self, E, H, transformed_E, sum_EH, P=None):
        if self.args.active_mode == "sprtrelu":
            if self.args.inter_dropout == 0.0:
                Z = H + torch.einsum("nd,bdc->bnc", transformed_E, sum_EH)
            elif self.args.inter_dropout == 1.0:
                Z_concat = torch.concat([torch.einsum("nm,bmc->bnc", torch.einsum("nd,dm->nm", transformed_E[self.args.nodes_per[cid],:], transformed_E[self.args.nodes_per[cid],:].T), H[:,self.args.nodes_per[cid],:])
                                      for cid in range(self.args.num_clients)], dim=1)
                Z = torch.zeros_like(Z_concat)
                Z[:, self.indices, :] = Z_concat
                Z = H + Z
        elif self.args.active_mode == "adptpolu":
            if self.args.inter_dropout == 0.0:
                Z = torch.stack([torch.einsum("nd,bdc->bnc", transformed_E[i], sum_EH[i]) for i in range(self.args.act_k+1)])
                Z_concat = torch.concat([torch.einsum("ak,kbnc->abnc", P[cid], Z[:,:,self.args.nodes_per[cid],:])[0]
                                for cid in range(self.args.num_clients)], dim=1)
                Z = torch.zeros_like(Z_concat)
                Z[:, self.indices, :] = Z_concat
                assert Z.shape == H.shape, (Z.shape, H.shape)
                Z = H + Z
            elif self.args.inter_dropout == 1.0:
                Z_concat = torch.concat([torch.stack([torch.einsum("nm,bmc->bnc", torch.einsum("nd,dm->nm", transformed_E[i][self.args.nodes_per[cid],:], transformed_E[i][self.args.nodes_per[cid],:].T), H[:,self.args.nodes_per[cid],:]) for i in range(self.args.act_k+1)])
                                  for cid in range(self.args.num_clients)], dim=2)
                Z = torch.zeros_like(Z_concat)
                Z[:, :, self.indices, :] = Z_concat
                Z_concat = torch.concat([torch.einsum("ak,kbnc->abnc", P[cid], Z[:,:,self.args.nodes_per[cid],:])[0]
                                for cid in range(self.args.num_clients)], dim=1)
                Z = torch.zeros_like(Z_concat)
                Z[:, :, self.indices, :] = Z_concat
                assert Z.shape == H.shape, (Z.shape, H.shape)
                Z = H + Z
        
        # Z = F.softmax(Z, dim=2)
        weights = [torch.einsum('nd,dio->nio', E[self.args.nodes_per[cid],...], self.weights_pool[cid]) for cid in range(self.args.num_clients)]  #N, dim_in, dim_out
        bias = [torch.matmul(E[self.args.nodes_per[cid],...], self.bias_pool[cid]) for cid in range(self.args.num_clients)]                       #N, dim_out
        x_gconv = [torch.einsum('bni,nio->bno', Z[:,self.args.nodes_per[cid],:], weights[cid]) + bias[cid] for cid in range(self.args.num_clients)]     #b, N, dim_out
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
