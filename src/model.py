import torch
#from src.causal_product import causal_dot_product_ref as causal_dot_product
from src.causal_product import causal_dot_product,causal_dot_product_ref


from src.mlp import MLP
from functools import partial

from multiprocessing.dummy import Pool

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

class CAMD(torch.nn.Module):
    def __init__(self, modality_dimensions, d_out, d_qk, n_layers=1, activation="relu", layernorm=False, skipconnections=False, skiptemperature=False):
        super(CAMD,self).__init__()
        self.M = len(modality_dimensions)
        self.d_v = d_out
        self.d_out = d_out
        self.modality_dimensions = modality_dimensions
        
        self.d_qk = d_qk
        self.feature_map = elu_feature_map
        self.W_Q = MLP(modality_dimensions["reference"], [d_qk]*n_layers, d_qk, activation,layernorm=layernorm, skipconnections=skipconnections, skiptemperature=skiptemperature)
        
        self.W_K = torch.nn.ModuleDict({mname:MLP(d_in, [d_qk]*n_layers, d_qk, activation,
                                            layernorm=layernorm, skipconnections=skipconnections, skiptemperature=skiptemperature) 
                for mname,d_in in modality_dimensions.items()})
        self.W_V = torch.nn.ModuleDict({mname:MLP(d_in, [self.d_v]*n_layers, self.d_v, activation,
                                            layernorm=layernorm, skipconnections=skipconnections, skiptemperature=skiptemperature) 
                for mname,d_in in modality_dimensions.items()})

        self.W_out = torch.nn.Linear(self.M*self.d_v , d_out)
        self.pool = None

    def forward(self, calX, pool=None):
        """
        calX is a dictionnary : {"reference":  shape (1,1,T_1,d_1), "m1":  shape (1,1,T_2,d_2), ...}
        """
        Q = self.feature_map(self.W_Q(calX["reference"]))
        
        t1 = calX["reference"][0, 0, :, -1]

        the_func = partial(self.forward_modality, t1=t1, Q=Q)
        if not (self.pool is None):
            results = self.pool.map(the_func, calX.items())
        else:
            results = list(map(the_func, calX.items()))
        
        # Concatenate on the head dimension
        Zout = torch.cat(results, dim=1)

        # Flatten all the heads
        #Zout = Zout.transpose(1,2).flatten(start_dim=2,end_dim=3)

        yhat = Zout.sum(1)#[...,:self.d_out]
        
        return yhat

    def forward_modality(self, args, t1=None, Q=None):
        #print(t1)
        modality_name, X = args
        K = self.feature_map(self.W_K[modality_name](X))
        V = self.W_V[modality_name](X)
        t2 = X[0,0,:,-1]
        causal_dot_product_func = causal_dot_product
        
        if t1.shape[0] == t2.shape[0]:
            #print("size(t1)=size(t2)=", t1.shape[0])
            causal_dot_product_func = causal_dot_product_ref
        out = causal_dot_product_func(Q, K, V, t1, t2)
        if out.isnan().any():
            print(modality_name,"NANs !!!")
        return out
