import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
limit_a, limit_b, epsilon = -.1, 1.1, 1e-6

class AGCRNCell(nn.Module):
    def __init__(self, args, adj, node_num, hyper_model_dim, dim_in, dim_out, cheb_k, embed_dim, weight_decay=1., droprate_init=0.5, temperature=2./3.,
                 lamba=1., local_rep=False, sparse= True):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.dim_in = dim_in
        self.args = args
        self.scale = 1e8

        if self.args.exp_mode == 'FED':
            if self.args.accelerate and self.args.inter_dropout in [0.0, 1.0]: from model.InterAGCN_fast import AVWGCN
            else: from model.InterAGCN_slow import AVWGCN
            self.gate = AVWGCN(self.args, dim_in+self.hidden_dim, 2*dim_out, embed_dim)
            self.update = AVWGCN(self.args, dim_in+self.hidden_dim, dim_out, embed_dim)
        else: 
            from model.AGCN import AVWGCNT
            self.gate = AVWGCNT(self.args, adj, dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
            self.update = AVWGCNT(self.args, adj, dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

        self.prior_prec = weight_decay
        self.qz_loga =None
        if self.args.exp_mode in ["CTR", "SGL"]:
            self.qz_linear = nn.Linear(hyper_model_dim, dim_in + self.hidden_dim)
        elif self.args.exp_mode == "FED":
            self.qz_linear = nn.ModuleList([nn.Linear(hyper_model_dim, dim_in + self.hidden_dim) for _ in range(args.num_clients)])
        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        self.lamba = lamba
        self.local_rep = local_rep
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor

    def forward(self, x, state, node_embeddings, poly_coefficients, h):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        if self.args.exp_mode in ["CTR", "SGL"]:
            self.qz_loga = self.qz_linear(h)
        elif self.args.exp_mode == "FED":
            qz_loga_per = [self.qz_linear[i](h[:, self.args.nodes_per[i], :]) for i in range(self.args.num_clients)]
            indices = [idx for sublist in self.args.nodes_per for idx in sublist]
            qz_loga_concat = torch.cat(qz_loga_per, dim=1)
            self.qz_loga = torch.zeros_like(qz_loga_concat)
            self.qz_loga[:, indices, :] = qz_loga_concat
        mask = self.sample_weights()
        total_sp, total_tran = 0, 0
        output, sp, tran = self.gate(input_and_state, node_embeddings, poly_coefficients, mask)
        total_sp += sp
        total_tran += tran
        z_r = torch.sigmoid(output)
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        output, sp, tran = self.update(candidate, node_embeddings, poly_coefficients, mask)
        total_sp += sp
        total_tran += tran
        hc = torch.tanh(output)
        h = r*state + (1-r)*hc
        normLoss = self.regularization(node_embeddings)
        return h, mask, normLoss, total_sp / 2, total_tran / 2

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)

    def fedavg(self):
        self.gate.fedavg()
        self.update.fedavg()

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self, node_embeddings):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        # supports = F.softmax(F.elu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        # logpw_col = torch.sum(- (.5 * self.prior_prec * supports.pow(2)) - self.lamba, 1)
        # print((1 - self.cdf_qz(0)).size(), logpw_col.size())
        # logpw = torch.sum((1 - self.cdf_qz(0)) * logpw_col)
        # #logpw = torch.sum(1 - self.cdf_qz(0))
        logpw = 0
        logpb = torch.sum(self.sample_weights())
        return logpw + logpb

    def regularization(self, node_embeddings):
        # return - (1. / (self.scale)*self._reg_w(node_embeddings))
        return (1. / (self.scale)*self._reg_w(node_embeddings))

    def count_expected_flops_and_l0(self):
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        # dim_in multiplications and dim_in - 1 additions for each output neuron for the weights
        # + the bias addition for each neuron
        # total_flops = (2 * in_features - 1) * out_features + out_features
        ppos = torch.sum(1 - self.cdf_qz(0))
        expected_flops = (2 * ppos - 1) * self.node_num
        expected_l0 = ppos * self.node_num
        
        return expected_flops.data[0], expected_l0.data[0]

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    def sample_weights(self):
        z = self.quantile_concrete(self.get_eps(self.floatTensor(self.args.num_nodes, self.dim_in + self.hidden_dim)))
        mask = F.hardtanh(z, min_val=0, max_val=1)
        #return mask.view(self.in_features, 1) * self.weights
        return mask
