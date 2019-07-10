import torch
from torch.autograd.variable  import Variable
import fused_dropout_add_cuda

class FusedDropoutAdd(torch.autograd.Function) :
    @staticmethod
    def forward(ctx, input, input_add, prob) :
        dropout_prob = 1. - prob
        output,mask = fused_dropout_add_cuda.forward(input, input_add, dropout_prob)
        scale = 1./dropout_prob
        scale_save = Variable(torch.tensor([scale]))
        ctx.save_for_backward(mask, scale_save);
        return output.detach()

    @staticmethod
    def backward(ctx, grad_output) :
        mask,scale = ctx.saved_tensors
        grad_input = fused_dropout_add_cuda.backward(grad_output, mask, scale[0])
        return grad_input, grad_output, None

fused_dropout_add = FusedDropoutAdd.apply
