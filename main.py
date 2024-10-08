import torch
import fast_transformers

from fast_transformers.attention import LinearAttention, CausalLinearAttention

from causal_product import causal_dot_product
from fast_transformers.causal_product import  causal_dot_product as causal_dot_product_reference


elu_feature_map = lambda x: torch.nn.functional.elu(x) + 1

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
    M = 10
    Tmax = 3
    Dmax = 2
    N = 1
    
    torch.manual_seed(0)
    Wref = torch.randn((Dmax, Dmax))
    Wnew = Wref.clone()#
    Wnew.requires_grad_(True)
    Wref.requires_grad_(True)

    # Create signals from M modalities, all with the same dimension and length, with irregular sampling
    D = torch.ones(M).long() * Dmax
    T = [Tmax]*M
    names = ["m{}".format(i+1) for i in range(M)]
    #timelines = {k: torch.sort(torch.randn(t).abs(),descending=False).values*100 for k,t in zip(names, T)}
    timelines = {k: torch.arange(t,dtype=torch.float)+10 for k,t in zip(names, T)}

    data = {k: torch.randn(N, t, d) for k,t,d in zip(names, T, D)}

    x1 = data["m1"]
    t1 = timelines["m1"].reshape(-1)-1

    x2 = data["m2"]
    t2 = timelines["m2"].reshape(-1)
    
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

    print("Queries:",queries[0,0])
    print("Keys:",keys[0,0])
    print("Values:",values[0,0])
    print("Tq:",  t1)
    print("Tkv:", t2)
    output = causal_dot_product(queries@Wnew, keys@Wnew, values, t1, t2)
    
    print("Output")
    print(output)
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

    ref_output = causal_dot_product_reference(queries@Wref, keys@Wref, values)
    print("Ref")
    print(ref_output)
    loss = (ref_output).sum()**2
    loss.backward()

    print("Backward")

    print(Wref.grad)

    pass
