import torch
from torch.autograd.variable  import Variable
import fused_relu_dropout_cuda

class FusedReluDropout(torch.autograd.Function) :
    @staticmethod
    def forward(ctx, input, prob) :
        dropout_prob = 1. - prob
        output,mask = fused_relu_dropout_cuda.forward(input, dropout_prob)
        scale = 1./dropout_prob
        scale_save = Variable(torch.tensor([scale]))
        ctx.save_for_backward(mask, scale_save);
        return output.detach()

    @staticmethod
    def backward(ctx, grad_output) :
        mask,scale = ctx.saved_tensors
        grad_input = fused_relu_dropout_cuda.backward(grad_output, mask, scale[0])
        return grad_input, None

fused_relu_dropout = FusedReluDropout.apply
