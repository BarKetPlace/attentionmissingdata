import torch
if torch.cuda.is_available():
    from src.causal_product import causal_dot_product
else:
    from src.causal_product import causal_dot_productCPU as causal_dot_product

from src.mlp import MLP
from functools import partial
import torch.multiprocessing as mp
from multiprocessing import Process, Queue


class CAMD(torch.nn.Module):
    def __init__(self, M, d_v, d_qk, d_out, n_layers, activation, layernorm=False, skipconnections=False, skiptemperature=False):
        super(CAMD,self).__init__()
        self.M = M
        self.d_v = d_v
        self.d_out = d_out
        self.d_qk = d_qk
        
        self.W_Q = MLP(d_qk + 2, [d_qk+ 2]*n_layers,d_qk+ 2,activation,layernorm=layernorm, skipconnections=skipconnections, skiptemperature=skiptemperature)
        
        self.W_K = [MLP(d_qk + 2, [d_qk+ 2]*n_layers,d_qk+ 2,activation,layernorm=layernorm, skipconnections=skipconnections, skiptemperature=skiptemperature) for _ in range(M)]

        self.W_out = torch.nn.Linear(M*d_v, d_out)
    
    def forward(self, calX):
        """
        calX is a dictionnary : {"m1":  shape (1,1,T_1,d_1), "m2":  shape (1,1,T_2,d_2), ...}
        """
        Q = self.W_Q(calX["m1"])
        the_queue = Queue()
        M = len(self.W_K)
        t1 = calX["m1"][0, 0, :, -1]

        Zout = torch.zeros((1, M, t1.shape[0], self.d_qk+ 2), device=Q.device).share_memory_()
        
        all_functions = [partial(forward_modality, m=m,W_k=self.W_K[m], t1=t1, Q=Q) for m in range(len(self.W_K))]
        outputs = []
        
        processes = []
        for func, X in zip(all_functions, calX.values()):
            p = mp.Process(target=func, args=(X, Zout))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()

        # Concatenate on the head dimension
        #Zout = torch.cat(Z_m, dim=1)
        
        # Flatten all the heads
        #Zout = Zout.transpose(1,2).flatten(start_dim=2,end_dim=3)

        yhat = Zout.sum(1)[...,:2]
        #yhat = self.W_out(Zout)
        return yhat

def forward_modality(X, output, m=None,W_k=None, t1=None, Q=None):
    K = W_k(X)
    V = X
    t2 = X[0,0,:,-1]
    #the_queue.put((m,causal_dot_product(Q, K, V, t1, t2)))
    if not (output is None):
        output[:,m,...] = causal_dot_product(Q, K, V, t1, t2)
        return None
    else:
        return causal_dot_product(Q, K, V, t1, t2)
