import torch
# a = torch.randn(4,4,7,8) #(batch,head,seq,k_dim)
# b = torch.randn(4,8,5,8) #(head,k_dim,bucket,round)
# print(torch.einsum('bhsk,hkvr->bhsvr',a,b).shape)

# a = torch.arange(24).reshape(2,3,4)
# b = torch.arange(60).reshape(3,4,5)
# print(a)
# print(b)
# c = torch.einsum('abc,bcd->abd',a,b)
# print(c)
# print(c.shape)
a = torch.randn(4,5,6)
test = a