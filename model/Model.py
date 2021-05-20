import torch
import torch.nn as nn
from Encoder import *
from CRF_Decoder import*
class Model(nn.Module):
    def __init__(self, config, src_embedding_num, tag_num, embedding_matrix, embedding_dim_size):
        super(Model,self).__init__()
        self.config = config
        self.encoder = Encoder(config, src_embedding_num, embedding_matrix, embedding_dim_size)
        self.decoder = CRF_decoder(config.d_model, tag_num)


    def get_encoder_output(self,src,mask,use_gpu):
        # faltten
        reverse_mask = (mask == False)
        src_ = torch.masked_select(src, reverse_mask)
        src_ = src_.unsqueeze(0)
        # 拿到 输出
        pad_idx = 1
        encoder_output,buckets = self.encoder(src_, pad_idx, use_gpu)
        feature_vec = torch.randn((reverse_mask.size()[0], reverse_mask.size()[1], self.config.d_model),device='cuda')
        sentence_len_tensor = torch.sum(reverse_mask, dim=-1).long()
        begin_offset = 0
        for i in range(sentence_len_tensor.shape[0]):
            feature_vec[i][0:sentence_len_tensor[i]] = encoder_output[:,begin_offset: begin_offset+ sentence_len_tensor[i],:]
            begin_offset += sentence_len_tensor[i]
        return feature_vec,buckets
    
    def forward(self, src, y, mask, use_gpu):
        """
        :param src:(当前分段句子个数，seq_len)
        :param y: (当前分段句子个数，seq_len)
        :param mask:(当前分段句子个数，seq_len)
        :param use_gpu:是否使用gpu
        :return:
        """ 
        feature_vec = self.get_encoder_output(src, mask, use_gpu)
        loss, path = self.decoder.loss(feature_vec, y, mask, use_gpu)
        return loss, path