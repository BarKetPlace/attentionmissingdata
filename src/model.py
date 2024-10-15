import torch
from src.causal_product import causal_dot_product_ref as causal_dot_product
from src.causal_product import causal_dot_product


from src.mlp import MLP
from functools import partial
import torch.multiprocessing as mp
from multiprocessing.dummy import Pool

#from multiprocessing import Process, Queue


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
        
    def forward(self, calX):
        """
        calX is a dictionnary : {"m1":  shape (1,1,T_1,d_1), "m2":  shape (1,1,T_2,d_2), ...}
        """
        Q = self.W_Q(calX["m1"])
        
        t1 = calX["m1"][0, 0, :, -1]

        Zout = torch.zeros((1, self.M, t1.shape[0], self.d_qk+ 2), device=Q.device).share_memory_()
        
        the_func = partial(forward_modality, t1=t1, Q=Q)

        with Pool(self.M) as pool:
            results = pool.map(the_func, zip(calX.values(), self.W_K))
        
        # Concatenate on the head dimension
        Zout = torch.cat(results, dim=1)

        # Flatten all the heads
        #Zout = Zout.transpose(1,2).flatten(start_dim=2,end_dim=3)

        yhat = Zout.sum(1)[...,:2]
        
        return yhat

def forward_modality(args, t1=None, Q=None):
    #print(t1)
    X, W_k = args
    K = W_k(X)
    V = X
    t2 = X[0,0,:,-1]
    return causal_dot_product(Q, K, V, t1, t2)
