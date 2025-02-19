import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

class AVWGCN(nn.Module):
    def __init__(self, args, adj, dim_in, dim_out, embed_dim):
        super(AVWGCN, self).__init__()
        self.args = args
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.adj = adj

        if self.args.exp_mode == 'SGL':
            total_nodes = self.args.num_nodes
            self.mask = torch.zeros(total_nodes, total_nodes).to(self.args.device)
            partitions = self.args.nodes_per
            for part in partitions:
                for i in range(len(part)):
                    for j in range(i, len(part)):
                        self.mask[part[i]][part[j]] = 1.0
                        self.mask[part[j]][part[i]] = 1.0

    def forward(self, x, node_embeddings, poly_coefficients):
        assert poly_coefficients == None
        assert self.args.active_mode in ['softmax', 'sprtrelu']
        
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]

        if self.args.active_mode == 'softmax':
            supports = torch.mm(node_embeddings, node_embeddings.transpose(0, 1))
            supports = F.softmax(F.relu(supports), dim=1)
        elif self.args.active_mode == 'sprtrelu':
            # supports = torch.mm(F.relu(node_embeddings), F.relu(node_embeddings.transpose(0, 1)))
            supports = torch.mm(F.softmax(F.relu(node_embeddings), dim=1), F.softmax(F.relu(node_embeddings.transpose(0, 1)), dim=1))

        support_set = [torch.eye(node_num).to(supports.device), supports]
        supports = sum(support_set) #N, N

        if self.args.exp_mode == 'SGL':
            supports = supports * self.mask
            # self.args.logger.info("apply mask!!!!!!")

        weights = torch.einsum('nd,dio->nio', node_embeddings, self.weights_pool)  #N, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("nm,bmc->bnc", supports, x)      #B, N, dim_in
        x_gconv = torch.einsum('bni,nio->bno', x_g, weights) + bias     #B, N, dim_out
        return x_gconv
    

class AVWGCNT(nn.Module):
    def __init__(self, args, adj, dim_in, dim_out, cheb_k, embed_dim,num_node, sparse= True):
        super(AVWGCNT, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.layernorm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        self.adj = adj
        self.num_node = num_node
        self.args = args
        
        if self.args.exp_mode == 'SGL':
            total_nodes = self.args.num_nodes
            self.mask = torch.zeros(total_nodes, total_nodes).to(self.args.device)
            partitions = self.args.nodes_per
            for part in partitions:
                for i in range(len(part)):
                    for j in range(i, len(part)):
                        self.mask[part[i]][part[j]] = 1.0
                        self.mask[part[j]][part[i]] = 1.0
       
    def forward(self, x, node_embeddings, poly_coefficients, mask =None):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        assert poly_coefficients == None
        assert self.args.active_mode in ['softmax', 'sprtrelu']
        
        batch_size = x.size(0)
        
        node_num = node_embeddings.shape[0]
        node_embeddings = self.dropout(self.layernorm(node_embeddings))

        if self.args.active_mode == 'softmax':
            supports = torch.mm(node_embeddings, node_embeddings.transpose(0, 1))
            supports = F.softmax(F.relu(supports), dim=1)
        elif self.args.active_mode == 'sprtrelu':
            # supports = torch.mm(F.relu(node_embeddings), F.relu(node_embeddings.transpose(0, 1)))
            supports = torch.mm(F.softmax(F.relu(node_embeddings), dim=1), F.softmax(F.relu(node_embeddings.transpose(0, 1)), dim=1))

        if self.args.exp_mode == 'SGL':
            supports = supports * self.mask
            # self.args.logger.info("apply mask!!!!!!")

        if mask is not None:
            #self.qz_loga = self.qz_linear(h)
            #mask = self.adj.to(x.device)*self.sample_weights()
            supports = supports*mask
            support_set = [torch.eye(node_num).expand(batch_size,node_num, node_num).to(supports.device), supports]
        else:
            support_set = [torch.eye(node_num).to(supports.device), supports]
            #*self.adj.to(x.device)
            mask = self.adj
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        if supports.size(1) != node_num:
            x_g = torch.einsum("kbnm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        else:
            x_g = torch.einsum("knm,bmc->bknc", supports, x)
        
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return x_gconv
    