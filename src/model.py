import torch
#from src.causal_product import causal_dot_product_ref as causal_dot_product
from src.causal_product import causal_dot_product,causal_dot_product_ref


from src.mlp import MLP
from functools import partial

from multiprocessing.dummy import Pool


class CAMD(torch.nn.Module):
    def __init__(self, M, d_v, d_qk, d_out, n_layers, activation, layernorm=False, skipconnections=False, skiptemperature=False):
        super(CAMD,self).__init__()
        self.M = M
        self.d_v = d_v
        self.d_out = d_out
        self.d_qk = d_qk
        
        self.W_Q = MLP(d_qk + 2, [d_qk+ 2]*n_layers,d_qk+ 2,activation,layernorm=layernorm, skipconnections=skipconnections, skiptemperature=skiptemperature)
        
        self.W_K = torch.nn.ModuleList([MLP(d_qk + 2, [d_qk+ 2]*n_layers,d_qk+ 2,activation,layernorm=layernorm, skipconnections=skipconnections, skiptemperature=skiptemperature) for _ in range(M)])

        self.W_out = torch.nn.Linear(M*d_v, d_out)
        self.pool = None

    def forward(self, calX,pool=None):
        """
        calX is a dictionnary : {"m1":  shape (1,1,T_1,d_1), "m2":  shape (1,1,T_2,d_2), ...}
        """
        Q = self.W_Q(calX["m1"])
        
        t1 = calX["m1"][0, 0, :, -1]

        the_func = partial(forward_modality, t1=t1, Q=Q)
        if not (self.pool is None):
            results = self.pool.map(the_func, zip(calX.values(), self.W_K))
        else:
            results = list(map(the_func, zip(calX.values(), self.W_K)))
        
        # Concatenate on the head dimension
        Zout = torch.cat(results, dim=1)

        # Flatten all the heads
        #Zout = Zout.transpose(1,2).flatten(start_dim=2,end_dim=3)

        yhat = Zout.sum(1)[...,:self.d_out]
        
        return yhat

def forward_modality(args, t1=None, Q=None):
    #print(t1)
    X, W_k = args
    K = W_k(X)
    V = X
    t2 = X[0,0,:,-1]
    causal_dot_product_func = causal_dot_product
    
    if t1.shape[0] == t2.shape[0]:
        print("size(t1)=size(t2)=", t1.shape[0])
        causal_dot_product_func = causal_dot_product_ref

    return causal_dot_product_func(Q, K, V, t1, t2)
