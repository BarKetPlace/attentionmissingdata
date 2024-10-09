import torch
from src.causal_product import causal_dot_product

from src.mlp import MLP

class CAMD(torch.nn.Module):
    def __init__(self, M, d_v, d_qk, d_out, n_layers, activation, layernorm=False, skipconnections=False, skiptemperature=False):
        super(CAMD,self).__init__()
        self.M = M
        self.d_v = d_v
        self.d_out = d_out
        self.d_qk = d_qk
        
        self.W_Q = MLP(d_qk + 2, [d_qk+ 2]*n_layers,d_qk+ 2,activation,layernorm=layernorm, skipconnections=skipconnections, skiptemperature=skiptemperature) ##torch.nn.Linear(d_qk + 2, d_qk)
        self.W_K = MLP(d_qk + 2, [d_qk+ 2]*n_layers,d_qk+ 2,activation,layernorm=layernorm, skipconnections=skipconnections, skiptemperature=skiptemperature) ##torch.nn.Linear(d_qk + 2, d_qk)
        #self.W_V = MLP(d_v + 2, [d_v]*n_layers,d_v,activation,   layernorm=layernorm, skipconnections=skipconnections, skiptemperature=skiptemperature)    ##torch.nn.Linear(d_v + 2, d_v)
        
        self.W_out = torch.nn.Linear(M*d_v, d_out)

    def forward(self, calX):
        """
        calX is a dictionnary : {"m1":  shape (1,1,T_1,d_1), "m2":  shape (1,1,T_2,d_2), ...}
        """
        Z_m = {}
        Q = self.W_Q(calX["m1"])

        t1 = calX["m1"][0, 0, :, -1]
        
        for k, X in calX.items():
            K = self.W_K(X)
            V = X

            t2 = X[0,0,:,-1]
            
            Z_m[k] = causal_dot_product(Q, K, V, t1, t2)
        
        # Concatenate on the head dimension
        Zout = torch.cat(list(Z_m.values()), dim=1)
        
        # Flatten all the heads
        #Zout = Zout.transpose(1,2).flatten(start_dim=2,end_dim=3)

        yhat = Zout.sum(1)[...,:2]
        #yhat = self.W_out(Zout)
        return yhat
