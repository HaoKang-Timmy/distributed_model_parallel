
from torch import autograd
import torch.distributed as dist
import torch
import torch.nn as nn

class ForwardSend_BackwardReceive(autograd.Function):
    @staticmethod
    def forward(ctx,input:torch.tensor,from_rank:int,to_rank:int,self_rank:int):
        ctx.save_for_backward(torch.tensor(from_rank),torch.tensor(to_rank),torch.tensor(self_rank))
        dist.send((torch.tensor(input.dim())*torch.tensor(1.0)).to(self_rank), to_rank)
        dist.send((torch.tensor(input.size())*torch.tensor(1.0)).to(self_rank), to_rank)
        dist.send(input, to_rank)
        #print("forward send",input.shape,"from",self_rank,"to",to_rank)
        return input
    @staticmethod
    def backward(ctx, grad_output):
        from_rank,to_rank,self_rank = ctx.saved_tensors
        dim = torch.tensor(1.0).cuda(int(self_rank))
        dist.recv(dim,int(from_rank))
        size = torch.rand(int(dim)).cuda(int(self_rank))
        dist.recv(size,int(from_rank))
        output = torch.rand(tuple(size.int())).cuda(int(self_rank))
        dist.recv(output,int(from_rank))
        #print("backward recv",output.shape,"from",from_rank,"to",self_rank)
        return output,None,None,None

# class ForwardReceive_BackwardSend_info(autograd.Function):
#     @staticmethod
#     def forward(ctx,input:torch.tensor,from_rank:int,to_rank:int,self_rank:int): 
#         dim = torch.tensor(1.0).cuda(self_rank)
#         dist.recv(dim,from_rank)
#         size = torch.rand(int(dim))
#         dist.recv(size,from_rank)
#         output = torch.rand(tuple(size.int())).cuda(self_rank)
#         return output
#     @staticmethod
#     def backward(ctx,grad_output):
#         return grad_output*1.0,None,None
def generate_recv(from_rank:int,self_rank:int):
    dim = torch.tensor(1.0).cuda(self_rank)
    dist.recv(dim,from_rank)
    size = torch.rand(int(dim)).cuda(self_rank)
    dist.recv(size,from_rank)
    output = torch.rand(tuple(size.int())).cuda(self_rank)
    output = output.requires_grad_()
    return output
class ForwardReceive_BackwardSend(autograd.Function):
    @staticmethod
    def forward(ctx,input:torch.tensor,from_rank:int,to_rank:int,self_rank:int):
        ctx.save_for_backward(torch.tensor(from_rank),torch.tensor(to_rank),torch.tensor(self_rank))
        dist.recv(input,from_rank)
        #print("forward recv",input.shape,"from",from_rank,"to",self_rank)
        return input*1.0
    @staticmethod
    def backward(ctx,grad_output):
        from_rank,to_rank,self_rank = ctx.saved_tensors
        dist.send((torch.tensor(grad_output.dim())*torch.tensor(1.0)).to(int(self_rank)), int(to_rank))
        dist.send((torch.tensor(grad_output.size())*torch.tensor(1.0)).to(int(self_rank)), int(to_rank))
        dist.send(grad_output, int(to_rank))
        #print("backward send",grad_output.shape,"from",self_rank,"to",to_rank)
        return grad_output,None,None,None

class Reshape(nn.Module):
    def __init__(self,*args):
        super(Reshape,self).__init__()
        self.shape =args
    def forward(self,x):
        return x.view(self.shape)



