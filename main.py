import torch
#import fast_transformers

#from fast_transformers.attention import LinearAttention, CausalLinearAttention
#from fast_transformers.feature_maps import elu_feature_map,ActivationFunctionFeatureMap


from causal_product import causal_dot_product

#relu_feature_map = ActivationFunctionFeatureMap.factory(
#    lambda x: torch.nn.functional.relu(x)
#)

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
    Tmax = 23
    Dmax = 4
    N = 1
    
    

    # Create signals from M modalities, all with the same dimension and length, with irregular sampling
    D = torch.ones(M).long() * Dmax
    T = [Tmax]*M
    names = ["m{}".format(i+1) for i in range(M)]
    timelines = {k: torch.sort(torch.randn(t).abs(),descending=False).values*100 for k,t in zip(names, T)}
    
    data = {k: torch.randn(N,t,d) for k,t,d in zip(names, T, D)}
    x1 = data["m1"]
    t1 = timelines["m1"].reshape(-1, 1)

    x2 = data["m2"]
    t2 = timelines["m2"].reshape(1, -1)
    
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
    output = causal_dot_product(queries, keys, values)
    print(output)

    # key_lengths = fast_transformers.masking.LengthMask(torch.tensor([Tmax]*N).long(), max_len=None, device=None)
    # query_lengths = fast_transformers.masking.LengthMask(torch.tensor([Tmax]*N).long(), max_len=None, device=None)
    
    ## Try linear attention with a full mask, i.e. non-causal attention
    #attn_mask = fast_transformers.masking.FullMask(Tmax)
    #lin_attn = LinearAttention(Dmax)
    #lin_attn(queries, keys, values, attn_mask, query_lengths, key_lengths)

    ## Try with a lower triangular mask, i.e. ignore irregular sampling just for testing the existing
    #attn_mask = fast_transformers.masking.TriangularCausalMask(Tmax)
    #causal_attn = CausalLinearAttention(Dmax)
    #causal_attn(queries, keys, values, attn_mask, query_lengths, key_lengths)

    pass
