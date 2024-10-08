import torch
import fast_transformers

from src.causal_product import causal_dot_product
from fast_transformers.causal_product import  causal_dot_product as causal_dot_product_reference

def test(M,Dmax,Tmax):
    N=1
    names = ["m{}".format(i+1) for i in range(M)]

    D = torch.ones(M).long() * Dmax
    #T = [Tmax] + torch.randint(Tmax//3,Tmax,(M-1,)).long().numpy().tolist()# [Tmax]*()
    T = [Tmax] + [Tmax]*(M-1)

    #timelines = {k: torch.sort(torch.randn(t).abs(),descending=False).values*100 for k,t in zip(names, T)}
    timelines = {k: torch.arange(t, dtype=torch.float)+10 for k,t in zip(names, T)}

    data = {k: torch.randn(N, t, d) for k,t,d in zip(names, T, D)}

    x1 = data["m1"]
    t1 = timelines["m1"].reshape(-1)

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

    print("Output")
    print(output)
    loss=(output**2).sum()
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

    print("Ref")
    print(ref_output)
    loss = (ref_output**2).sum()
    loss.backward()

    print("Backward")
    print(Wref.grad)

    pass
