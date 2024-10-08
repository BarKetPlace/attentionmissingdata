#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

import torch

from .causal_product_numerator_cpu import causal_dot_numerator_product as causal_dot_numerator_product_cpu, \
                                 causal_dot_numerator_backward as causal_dot_numerator_backward_cpu

#from .causal_product_denominator_cpu import causal_dot_denominator_product as causal_dot_denominator_product_cpu, \
#                                 causal_dot_denominator_backward as causal_dot_denominator_backward_cpu


try:
    from .causal_product_cuda import \
        causal_dot_product as causal_dot_product_cuda, \
        causal_dot_backward as causal_dot_backward_cuda
except ImportError:
    causal_dot_product_cuda = causal_dot_backward_cuda = None


from fast_transformers.causal_product import  causal_dot_product as causal_dot_product_reference


def causal_dot_product(Q, K, V, tq, tkv):
    product = causal_dot_numerator_product(Q, K, V, tq, tkv)
    N, H, L = V.shape[:-1]
    Vdummy = torch.ones((N, H, L, 1), device=V.device)

    normalization = causal_dot_numerator_product(Q, K, Vdummy, tq, tkv)
    return product / (normalization + 1e-6)

class CausalDotProductNumerator(torch.autograd.Function):
    """Compute the weighted sum of values but attending only to previous
    values."""
    dot_numerator = {
        "cpu": causal_dot_numerator_product_cpu,
        "cuda": causal_dot_product_cuda
    }
    dot_numerator_backward = {
        "cpu": causal_dot_numerator_backward_cpu,
        "cuda": causal_dot_backward_cuda
    }

    @staticmethod
    def forward(ctx, Q, K, V, tq, tkv):
        # Save the inputs for the gradient computation
        ctx.save_for_backward(Q, K, V, tq, tkv)

        # Create the output tensor
        device = Q.device
        N, H, L, _ = Q.shape
        _, _, _, M = V.shape
        product = torch.zeros((N, H, L, M), device=device)

        # Actually perform the numerator of dot product
        CausalDotProductNumerator.dot_numerator[device.type](
            Q.data,
            K.data,
            V.data,
            tq,
            tkv,
            product
        )

        product_ref = causal_dot_product_reference(Q, K, V)
        return product

    @staticmethod
    def backward(ctx, grad_out):
        # Extract the saved tensors
        Q, K, V, tq, tkv = ctx.saved_tensors

        # Allocate memory for the gradients
        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)
        grad_tq = torch.zeros_like(tq)
        grad_tkv = torch.zeros_like(tkv)

        # Actually compute the gradients
        CausalDotProductNumerator.dot_numerator_backward[Q.device.type](
            Q.data,
            K.data,
            V.data,
            tq, tkv,
            grad_out,
            grad_Q,
            grad_K,
            grad_V
        )
        #print("Numerator grad_Q", grad_Q)
        return grad_Q, grad_K, grad_V, grad_tq, grad_tkv



# Alias the autograd functions to python style snake case naming
causal_dot_numerator_product = CausalDotProductNumerator.apply
