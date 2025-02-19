from re import X
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0,embed_dim,2).float()*-(math.log(10000.0)/embed_dim)).exp()
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self,x):
        return self.pe[:,:x.size(1)].unsqueeze(2).expand_as(x).detach()

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        # print(x.shape)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    """
    DLinear
    """
    def __init__(self,hyper_source_len, rnn_units, num_nodes):
        super(DLinear, self).__init__()
        self.seq_len = hyper_source_len
        self.pred_len = rnn_units

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = True
        self.channels =num_nodes

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            self.Linear_Decoder = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                self.Linear_Decoder.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Decoder = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x
        

class HyperNet(nn.Module):
    def __init__(self, hyper_source_len, rnn_units, embed_dim, node_num):
        super(HyperNet, self).__init__()
        
        self.position = PositionalEncoding(rnn_units)
        self.avgpool = nn.AvgPool1d(12,12)
        self.dlinear = DLinear(hyper_source_len, rnn_units, node_num)
        self.xlinear =nn.Linear(1, rnn_units)
        self.rnn_units = rnn_units
        self.corss_attention = nn.Transformer(d_model = rnn_units, 
                                              nhead = 4, 
                                              num_encoder_layers= 2, 
                                              num_decoder_layers= 2, 
                                              dim_feedforward= 4*rnn_units, 
                                              dropout=0.1, 
                                              activation='gelu',
                                              batch_first=True, 
                                              norm_first=True)
                                              
    def forward(self, hyper_source, source):    #source shaped: B,L,N,1
        hyper_source = hyper_source.squeeze()
        batch_size, horizon, node_nums,_ = source.size()
        
        hyper_source= self.avgpool(hyper_source.transpose(-1,-2)).transpose(-1,-2)
        
        hyper_emb = self.dlinear(hyper_source) #shaped: source_nums, node_nums, dim 
        hyper_emb = hyper_emb.unsqueeze(0)
        source = self.xlinear(source) #source shape:B,L,N,D
        source +=self.position(source)
        hyper_emb += self.position(hyper_emb)
        #hyper_emb =hyper_emb.transpose(0,1)
        HL = hyper_emb.size(1)
        
        B,L,N,D = source.size()
        hyper_emb = hyper_emb.expand(B,HL,N,D)
        hyper_emb = hyper_emb.transpose(0,3).reshape(N*B,HL,D)
        source= source.transpose(0,3).reshape(N*B,L,D)
       
        #hyper_emb = self.position(hyper_emb.unsqueeze(0))+hyper_emb
        
        
        
       
        
       
        outputs =self.corss_attention(hyper_emb, source) #shaped:L,N,D
        
        
        return outputs.view(B,N,L,D)
        
   
        
