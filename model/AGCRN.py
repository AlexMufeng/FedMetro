import torch, copy
import torch.nn as nn
import time
from model.AGCRNCell import AGCRNCell
import lib.utils as utils
from model.HYPER import HyperNet

class AGCRN(nn.Module):
    def __init__(self, args, adj):
        super(AGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.args = args

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        if self.args.active_mode == "adptpolu":
            self.poly_coefficients = nn.Parameter(torch.randn(args.num_clients, 1, args.act_k+1), requires_grad=True)
        else: self.poly_coefficients = None

        self.encoder = AVWDCRNN(self.args, adj, args.num_nodes, args.hyper_model_dim, args.input_dim, args.rnn_units,
                                args.cheb_k, args.embed_dim, args.num_layers)

        if self.args.exp_mode in ["CTR", "SGL"]:
            self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
            self.hypernet = HyperNet(int(args.hyper_horizon/12), args.rnn_units, args.embed_dim, args.num_nodes)
            self.main_weights_pool = nn.Parameter(torch.FloatTensor(args.embed_dim, args.rnn_units, args.input_dim))
        elif self.args.exp_mode == "FED":
            self.end_conv = nn.ModuleList([nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True) for _ in range(args.num_clients)])
            self.hypernet = nn.ModuleList([HyperNet(int(args.hyper_horizon/12), args.rnn_units, args.embed_dim, len(args.nodes_per[i]))for i in range(args.num_clients)])
            self.main_weights_pool = nn.ParameterList([torch.FloatTensor(args.embed_dim, args.rnn_units, args.input_dim) for _ in range(args.num_clients)])
       
    def forward(self, hyper_source, source):
        #source: B, T_1, N, D
        # start_time = time.time()  # 记录开始时间
        init_state = self.encoder.init_hidden(source.shape[0])
        
        if self.args.exp_mode in ["CTR", "SGL"]:
            h = self.hypernet(hyper_source, source)
            weights_h = torch.einsum('nd,dhi->nhi', self.node_embeddings, self.main_weights_pool) #N,hidden,I
        elif self.args.exp_mode == "FED":
            h_per = [self.hypernet[i](hyper_source[:, :, self.args.nodes_per[i], :], source[:, :, self.args.nodes_per[i], :]) for i in range(self.args.num_clients)]
            weights_hper = [torch.einsum('nd,dhi->nhi', self.node_embeddings[self.args.nodes_per[i], :], self.main_weights_pool[i]) for i in range(self.args.num_clients)]
            indices = [idx for sublist in self.args.nodes_per for idx in sublist]
            h_concat = torch.concat(h_per, dim=1)
            weights_concat = torch.concat(weights_hper, dim=0)
            h = torch.zeros_like(h_concat)
            weights_h = torch.zeros_like(weights_concat)
            h[:, indices, :, :] = h_concat
            weights_h[indices, ...] = weights_concat
        source = torch.cat([torch.einsum('bnld, ndi->blni',h,weights_h), source],-1)
        
        output, _, masks, normLoss, sp, tran = self.encoder(source, init_state, self.node_embeddings, self.poly_coefficients, h)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden

        #CNN based predictor
        if self.args.exp_mode in ["CTR", "SGL"]:
            output = self.end_conv(output)                         #B, T*C, N, 1
        elif self.args.exp_mode == "FED":
            output_per = [self.end_conv[i](output[:, :, self.args.nodes_per[i], :]) for i in range(self.args.num_clients)]
            output_concat = torch.concat(output_per, dim=2) #B, T*C, N, 1
            output = torch.zeros_like(output_concat)
            output[:, :, indices, :] = output_concat

        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C
        # end_time = time.time()  # 记录结束时间
        # elapsed_time = end_time - start_time  # 计算时间
        # print("Forward pass time: {:.4f} seconds".format(elapsed_time))
        return output, masks, normLoss, sp, tran

    def fedavg(self):
        if self.args.active_mode == "adptpolu":
            mean_p = torch.sum(self.poly_coefficients, dim=0, keepdim=True).data / self.args.num_clients
            self.poly_coefficients = nn.Parameter(torch.repeat_interleave(mean_p, self.args.num_clients, dim=0), requires_grad=True).to(mean_p.device)

        mean_w = utils.avg([conv.state_dict() for conv in self.end_conv])
        for i in range(self.args.num_clients): self.end_conv[i].load_state_dict(copy.deepcopy(mean_w))
        self.encoder.fedavg()


class AVWDCRNN(nn.Module):
    def __init__(self, args, adj, node_num, hyper_model_dim, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        self.node_num = node_num
        self.input_dim = dim_in * 2
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.args = args
        self.dcrnn_cells.append(AGCRNCell(self.args, adj, node_num, hyper_model_dim, self.input_dim, dim_out, cheb_k, embed_dim))

        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(self.args, adj, node_num, hyper_model_dim, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings, poly_coefficients, h):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim, (x.shape, self.node_num, self.input_dim)
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        mask_list = []
        normLoss_list = []
        total_sp, total_tran = 0, 0
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state, mask, normLoss, sp, tran = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings, poly_coefficients, h[:, :, t, :])
                inner_states.append(state)
                mask_list.append(mask)
                normLoss_list.append(normLoss)
                total_sp += sp
                total_tran += tran
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden, torch.stack(mask_list), torch.sum(torch.stack(normLoss_list)), total_sp / seq_length, total_tran / seq_length

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states
    
    def fedavg(self):
        for model in self.dcrnn_cells: model.fedavg()