import torch
import fast_transformers
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

from fast_transformers.attention import LinearAttention, CausalLinearAttention

from src.causal_product import causal_dot_product

elu_feature_map = lambda x: torch.nn.functional.elu(x) + 1
from fast_transformers.causal_product import  causal_dot_product as causal_dot_product_reference

from src.dataset import compute_target, prep_data, plot_data
from src.model import CAMD
from test import test

if __name__ == "__main__":
    M = 3
    Tmax = 5
    Dmax = 2
    N = 1
    d_out = 2
    #torch.manual_seed(0)
    
    names = ["m{}".format(i+1) for i in range(M)]

    # Create signals from M modalities, all with the same dimension and length, with irregular sampling
    D = torch.ones(M).long() * Dmax
    T = [Tmax] + torch.randint(2, Tmax, (M-1,)).long().numpy().tolist()
    
    timelines = {k: torch.sort(torch.randn(t).abs(), descending=False).values*100 for k,t in zip(names, T)}
    
    #timelines = {k: torch.arange(t, dtype=torch.float) for k,t in zip(names, T)}
    data = {k: torch.rand(N, t, d) for i,(k,t,d) in enumerate(zip(names, T, D))}

    y = compute_target(data, timelines, d_out)
    
    model =  CAMD(M, Dmax, Dmax, Dmax, n_layers=10, activation="relu", layernorm=True, skipconnections=True, skiptemperature=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    L = []

    X = prep_data(data, timelines)
    #fig, ax,images = plot_data(data,timelines,target=y[0],dim=0,figsize=(15,5))
    
    num_epochs = 2000
    every_e = num_epochs // 20
    for epoch in range(num_epochs):
        yhat = model(X)

        loss = torch.nn.functional.mse_loss(yhat, y)
        loss.backward()

        optimizer.step()
        
        L.append(loss.item())
        if (epoch % every_e) == 0:
            print(epoch, L[-1])
    
    fig, ax, images = plot_data(data,timelines,target=y[0],prediction=yhat[0].detach(),dim=0,figsize=(15,5),masks=False)
    plt.show()
    print("")
