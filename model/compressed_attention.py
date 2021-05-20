import torch
import torch.nn as nn
import torch.nn.functional as F

# 以下实现仅为 single-batch版本
# l 代表 local m 代表 conv
class Compressed_Attention(nn.Module):
    def __init__(self,name='l',head=4,k_dim=64,d_model=256,p_drop=0.1,compression_rate=4,block_size=64):
        super().__init__()
        self.name = name
        self.head = head
        self.d_model = d_model
        self.k_dim = k_dim
        self.block_size = block_size
        self.multi_head_query = nn.Linear(d_model,d_model)
        self.multi_head_key = nn.Linear(d_model,d_model)
        self.multi_head_value = nn.Linear(d_model,d_model)
        self.output_matrix = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(p_drop)
        if name == 'M':
            self.conv = nn.Conv1d(in_channels=d_model,out_channels=d_model,
                                kernel_size=compression_rate,stride=compression_rate,groups=head)
        
    def forward(self,src,input_mask=None,use_gpu=True):
        # src (batch_size,seq_len,d_model)
        # 首先判断当前模型是什么
        batch_size = src.size()[0]
        seq_len = src.size()[1]
        if self.name == 'L':
            if input_mask is not None:
                input_mask = input_mask.reshape((batch_size, -1, self.block_size))
            query = self.multi_head_query(src).view(batch_size, seq_len, self.head, -1)
            key = self.multi_head_key(src).view(batch_size, seq_len, self.head, -1)
            value = self.multi_head_value(src).view(batch_size, seq_len, self.head, -1)
            chunk_query = query.reshape(batch_size, -1, self.block_size, self.head, query.size()[-1])
            chunk_key = key.reshape(batch_size,-1,self.block_size,self.head,key.size()[-1])
            chunk_value = value.reshape(batch_size,-1,self.block_size,self.head,value.size()[-1])
            chunk_query = chunk_query.permute(0,3,1,2,4)
            chunk_key = chunk_key.permute(0,3,1,4,2)
            chunk_value = chunk_value.permute(0,3,1,2,4)
            #(batch,head,num,block_size,block_size)
            chunk_qk = chunk_query.matmul(chunk_key)* (self.k_dim ** -0.5)
            if input_mask is not None:
                chunk_qk = chunk_qk.masked_fill(input_mask[:,None,:,None,:],value=-1e10)
            chunk_qk = F.softmax(chunk_qk,dim=-1)
            chunk_qkv = torch.matmul(chunk_qk,chunk_value)
            #(batch,head,seq_len,k_dim)
            qkv = chunk_qkv.contiguous().view(batch_size,self.head,-1,chunk_qkv.size()[-1])
        if self.name =='M':
            multi_query = self.multi_head_query(src)
            multi_key = self.multi_head_key(src)
            multi_value = self.multi_head_value(src)
            # 因为这里含有pad,我们首先缩短 key value 的长度
            if input_mask is not None:
                pad_len = torch.sum(input_mask,dim=-1).item()
                multi_key = multi_key[:,:-pad_len,:]
                multi_value = multi_value[:,:-pad_len,:]
            multi_key = multi_key.transpose(1,2)
            multi_value = multi_value.transpose(1,2)
            #(batch,short_len,d_model)
            after_cnn_multi_key = self.conv(multi_key).transpose(1,2)
            after_cnn_multi_value = self.conv(multi_value).transpose(1,2)
            multi_query = multi_query.view(batch_size,-1,self.head,self.k_dim).permute(0,2,1,3)
            after_cnn_multi_key = after_cnn_multi_key.contiguous().view(batch_size,-1,self.head,self.k_dim).permute(0,2,3,1)
            after_cnn_multi_value = after_cnn_multi_value.contiguous().view(batch_size,-1,self.head,self.k_dim).permute(0,2,1,3)
            global_qk = torch.matmul(multi_query, after_cnn_multi_key)
            global_qk = F.softmax(global_qk,dim=-1)
            #(batch,head,seq_len,k_dim)
            qkv = torch.matmul(global_qk, after_cnn_multi_value)
        # 然后紧接着，
        qkv = qkv.permute(0,2,1,3).contiguous().view(batch_size,seq_len,-1)
        qkv = self.output_matrix(qkv)
        qkv = self.dropout(qkv)
        return qkv











