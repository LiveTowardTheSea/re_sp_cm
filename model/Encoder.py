import torch
import torch.nn as nn
import torch.nn.functional as F
from EncoderLayer import *
import numpy as np

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, config, src_embedding_num, embedding_matrix, embedding_dim_size):
        super(Encoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(src_embedding_num, embedding_dim_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.encoder_layer = nn.ModuleList([EncoderLayer(config, i) for i in range(config.layer_num)])

    def forward(self, src, pad_idx, use_gpu):
        # 对于 reformer 来说，输入必须可以整除 bucket_num
        # 对于 sparse 来说，输入长度必须可以整除 stride
        # 对于 compress 来说，输入必须可以整除 block_size
        batch_size = src.size()[0] # batch 数目
        seq_len = src.size()[1] # 原始输入长度
        buckets_list = np.zeros((self.config.layer_num, batch_size, self.config.head,self.config.round,seq_len)).astype(int)
        if self.config.name == 'reformer':
            divider = self.config.bucket
        elif self.config.name == 'sparse':
            divider = self.config.stride
        elif self.config.name == 'compress':
            divider = self.config.block_size
        
        pad_num = divider - seq_len % divider
        pad_vec = torch.full((batch_size,pad_num), fill_value=pad_idx, dtype=torch.long)
        if use_gpu:
            pad_vec = pad_vec.cuda()
        src_ = torch.cat([src,pad_vec],dim=-1)
        input_mask = torch.zeros((batch_size,src_.size()[-1]))
        input_mask[:,-pad_num:] += 1.0
        input_mask = input_mask.bool()
        if use_gpu:
            input_mask = input_mask.cuda()
        
        state = self.embedding(src_)
        for i in range(self.config.layer_num):
            state,buckets = self.encoder_layer[i](state,input_mask=input_mask,use_gpu=use_gpu)
            buckets_list[i] = buckets[:,:,:,0:seq_len]
        
        return state[:,:seq_len,:], buckets_list.squeeze(1)





        
