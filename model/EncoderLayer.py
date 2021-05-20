import torch
import torch.nn as nn
from LSH_Attention import *
from sparse_attention import *
from compressed_attention import *
from SubLayer import *

class EncoderLayer(nn.Module):
    def __init__(self,config,i):
        super().__init__()
        if config.name=='reformer':
            self.attention = LSHAttention(round_num=config.round, head=config.head, k_dim=config.k_dim,
                                            d_model=config.d_model, bucket_num=config.bucket, attend_chunk_num=config.attend,p_drop=config.p_drop)
        elif config.name == 'sparse':
            self.attention = Sparse_Attention(d_model=config.d_model,head=config.head,stride=config.stride,
                                                local_head_num=config.local_head,k_dim=config.k_dim,c=config.c,p_drop=config.p_drop)
        elif config.name == 'compress':
            self.attention = Compressed_Attention(name=config.layer[i],head=config.head,k_dim=config.k_dim,
                                                    d_model=config.d_model,p_drop=config.p_drop,
                                                    compression_rate=config.compression, block_size=config.block_size)
        self.atten_norm = LayerNorm(config.d_model)
        self.fcnn = Position_Wise_Network(config.d_model,config.d_ff,config.p_drop)
        self.fcnn_norm = LayerNorm(config.d_model)
    
    def forward(self, src, input_mask=None, use_gpu=True):
        # 添加 bucket为了reformer,不要就去掉咯
        temp,buckets = self.attention(src,input_mask=input_mask,use_gpu=use_gpu)
        src_ = src + temp
        attn = self.atten_norm(src_)
        temp = self.fcnn(attn)
        src_ =  attn + temp
        state = self.fcnn_norm(src_)
        return state,buckets
        

