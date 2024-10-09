import torch
import fast_transformers
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt


from fast_transformers.attention import LinearAttention, CausalLinearAttention

from causal_product import causal_dot_product


elu_feature_map = lambda x: torch.nn.functional.elu(x) + 1
from fast_transformers.causal_product import  causal_dot_product as causal_dot_product_reference

def test(D,T,names,Dmax):

    timelines = {k: torch.sort(torch.randn(t).abs(),descending=False).values*100 for k,t in zip(names, T)}
    #timelines = {k: torch.arange(t, dtype=torch.float)+10 for k,t in zip(names, T)}

    data = {k: torch.randn(N, t, d) for k,t,d in zip(names, T, D)}

    x1 = data["m1"]
    t1 = timelines["m1"].reshape(-1)-1

    x2 = data["m2"]
    t2 = timelines["m2"].reshape(-1)
    
    Wref = torch.randn((Dmax, Dmax))
    Wnew = Wref.clone()#
    Wnew.requires_grad_(True)
    Wref.requires_grad_(True)

    query = x1
    # Number of heads=1, 
    queries = query.unsqueeze(1)
    
    key = x2
    # Number of heads=1
    keys = key.unsqueeze(1)

    value = x2
    # Number of heads=1
    values = value.unsqueeze(1)

    # queries,keys and values of shape (N, h, T, d)
    print("\n".join([str(d.shape) for d in [queries@Wnew, keys@Wnew, values, t1, t2]]))

    #print("Queries:",queries[0,0]@Wnew)
    #print("Keys:",keys[0,0]@Wnew)
    #print("Values:",values[0,0]@Wnew)
    print("Tq:",  t1)
    print("Tkv:", t2)
    print(t1.reshape(-1,1)>=t2.reshape(1,-1))
    #output = causal_dot_product(queries@Wnew, keys@Wnew, values, t1, t2)
    
    Q = queries@Wnew
    K = keys@Wnew
    V = values
    output = causal_dot_product(Q, K, V, t1, t2)
    
   # N, H, L = V.shape[:-1]
    #Vdummy = torch.ones((N, H, L, 1), device=V.device)

    #normalization = causal_dot_numerator_product(Q, K, Vdummy, t1, t2)
    #output = product / (normalization+1e-6)

    #print("Output")
    #print(output)
    loss=(output).sum()**2
    loss.backward()

    print("Backward")
    print(Wnew.grad)

    key_lengths = fast_transformers.masking.LengthMask(torch.tensor([Tmax]*N).long(), max_len=None, device=None)
    query_lengths = fast_transformers.masking.LengthMask(torch.tensor([Tmax]*N).long(), max_len=None, device=None)
    
    ## Try linear attention with a full mask, i.e. non-causal attention
    #attn_mask = fast_transformers.masking.FullMask(Tmax)
    #lin_attn = LinearAttention(Dmax)
    #lin_attn(queries, keys, values, attn_mask, query_lengths, key_lengths)

    ## Try with a lower triangular mask, i.e. ignore irregular sampling just for testing the existing
    Q = queries@Wref
    K = keys@Wref

    product_ref = causal_dot_product_reference(Q, K, V)

    def norm_ref(Q,K):
        return torch.einsum("nhli,nhli->nhl", Q, K.cumsum(2)).unsqueeze(-1)
    normalization_ref = norm_ref(Q,K)
    
    ref_output = product_ref / (normalization_ref+1e-6)

    # Compute the normalizers

    #print("Ref")
    #print(ref_output)
    loss = (ref_output).sum()**2
    loss.backward()

    print("Backward")
    print(Wref.grad)

    pass

def linear_scaled_dot_product(queries, keys, values, feature_map, attn_mask=None, eps=1e-6):
    _,Tq,num_heads,d = queries.shape
    _,Tk,num_heads,d = keys.shape
    
    t1,t2 = attn_mask
    Y = torch.zeros_like(queries)
    Q = feature_map(queries)##.view(*queries.shape[:2],self.num_heads,self.key_dim)#[:,:,None,:]
    K = feature_map(keys)#.view(*keys.shape[:2],self.num_heads,self.key_dim) #[:,:,None,:]
    values = values#.view(*values.shape[:2],self.num_heads,self.out_dim)

    #for i in Tq:
        
    # Compute the KV matrix, namely the dot product of keys and values so
    # that we never explicitly compute the attention matrix and thus
    # decrease the complexity
    KV = torch.einsum("nthd,nthm->nhmd", K, values)

    # Compute the normalizer
    #Zi = 
    Z = 1/(torch.einsum("nthd,nhd->nth", Q, K.sum(dim=1))+eps)

    # Finally compute and return the new values
    V = torch.einsum("nthd,nhmd,nth->nthm", Q, KV, Z)
    V = V.flatten(start_dim=2,end_dim=3)
    return V

if __name__ == "__main__":
    M = 3
    Tmax = 10
    Dmax = 2
    N = 1
    d_out = 2
    torch.manual_seed(0)

    # Create signals from M modalities, all with the same dimension and length, with irregular sampling
    D = torch.ones(M).long() * Dmax
    T = [Tmax]*M
    names = ["m{}".format(i+1) for i in range(M)]

    #test(D,T,names,Dmax)

    #timelines = {k: torch.sort(torch.randn(t).abs(),descending=False).values*100 for k,t in zip(names, T)}
    #timelines = {k: torch.arange(t, dtype=torch.float)+10 for k,t in zip(names, T)}

    #data = {k: torch.randn(N, t, d) for k,t,d in zip(names, T, D)}

    timelines = {k: torch.sort(torch.randn(t).abs(), descending=False).values*100 for k,t in zip(names, T)}
    #timelines = {k: torch.arange(t, dtype=torch.float)+10 for k,t in zip(names, T)}
    
    data = {k: torch.randn(t, d)+i for i,(k,t,d) in enumerate(zip(names, T, D))}
    y = torch.zeros(Tmax, d_out)

    time_ref = timelines["m1"]
    for itime,t in enumerate(time_ref):
        x_max_previous = 0.
        for k in data.keys():
            previous_data = data[k][timelines[k] <= t]
            if previous_data.shape[0]>0:
                x_max_previous += data[k][timelines[k] <= t].max(0).values
        y[itime,:] = x_max_previous

    def prep_data(data,timelines):
        # Compute timeseries deltas
        deltas = {k: torch.diff(t, prepend=torch.tensor([t[0]])).view(t.shape[0],1) for k,t in timelines.items()}

        # Concatenate data and timelines
        calX = {k: torch.cat([data[k], deltas[k], timelines[k].view(-1,1)], dim=1).unsqueeze(0).unsqueeze(0) for k in data.keys()}
        return calX

    class CAMD(torch.nn.Module):
        def __init__(self, M, d_v, d_qk, d_out):
            super(CAMD,self).__init__()
            self.M = M
            self.d_v = d_v
            self.d_out = d_out
            self.d_qk = d_qk

            self.W_Q = torch.nn.Linear(d_qk + 2, d_qk)
            self.W_K = torch.nn.Linear(d_qk + 2, d_qk)
            self.W_V = torch.nn.Linear(d_v + 2, d_v)
            self.W_out = torch.nn.Linear(M*d_v, d_out)

        def forward(self, calX):
            """
            calX is a dictionnary : {"m1":  shape (1,1,T_1,d_1), "m2":  shape (1,1,T_2,d_2), ...}
            """
            Z_m = {}
            Q = self.W_Q(calX["m1"])

            t1 = calX["m1"][0,0,:,-1]
            
            for k, X in calX.items():
                K = self.W_K(X)
                V = self.W_V(X)

                t2 = X[0,0,:,-1]
                
                Z_m[k] = causal_dot_product(Q, K, V, t1, t2)
            
            # Concatenate on the head dimension
            Zout = torch.cat(list(Z_m.values()), dim=1)
            
            # Flatten all the heads
            Zout = Zout.transpose(1,2).flatten(start_dim=2,end_dim=3)

            # 
            yhat = self.W_out(Zout)
            return yhat
    
    model =  CAMD(M, Dmax, Dmax, Dmax)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

    x1 = data["m1"]
    t1 = timelines["m1"]
    L = []

    def plot_data(data, timelines, target, prediction=None, dim=0):
        fig, ax = plt.subplots()
        for i,k in enumerate(data.keys()):
            X = data[k][:,dim]
            timel = timelines[k]
            ax.plot(timel,X,"-",label=k,marker='o')
        ax.plot(timelines["m1"],target[:,dim],linewidth=2,marker="o",color="black",label="Target")
        if not (prediction is None):
            ax.plot(timelines["m1"],prediction[:,dim],linewidth=2,marker="o",color="darkred",label="Prediction")

        ax.legend()

        return fig, ax

    #fig, ax = plot_data(data,timelines,y)
    
    X = prep_data(data, timelines)

    num_epochs = 3000
    for epoch in range(num_epochs):
        yhat = model(X)

        loss = torch.nn.functional.mse_loss(yhat[0], y)
        loss.backward()
        optimizer.step()
        L.append(loss.item())
        print(epoch, L[-1])
    
    fig, ax = plt.subplots()
    ax.plot(L)

    fig, ax = plot_data(data, timelines, y, prediction=yhat.detach()[0], dim=0)
    plt.show()
