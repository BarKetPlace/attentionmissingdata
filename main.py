import torch
import os, sys

#os.environ['CUDA_LAUNCH_BLOCKING']="1"
#os.environ['TORCH_USE_CUDA_DSA'] = "1"
#import fast_transformers
#import matplotlib
#matplotlib.use("tkagg")
import matplotlib.pyplot as plt

#from fast_transformers.attention import LinearAttention, CausalLinearAttention

elu_feature_map = lambda x: torch.nn.functional.elu(x) + 1


from src.dataset import compute_target, prep_data, plot_data, compute_target4 as compute_target
from src.model import CAMD
from test import test
import pandas as pd

def get_data(thetype="synthetic"):
    if thetype == "robot":
        # https://archive.ics.uci.edu/dataset/963/ur3+cobotops
        fname = "data/dataset_02052023.xlsx"
        df = pd.read_excel(fname).drop(columns=["Num"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"].apply(lambda s: s.replace("\"","")),format="%Y-%m-%dT%H:%M:%S.%f%z")
        df.set_index("Timestamp",inplace=True)
        M = 6
        colnames = {"m{}".format(m):[s for s in df.columns if s.endswith(str(m))] for m in range(M)}
        X = {m: df[cols].copy() for m,cols in colnames.items()}
        df.drop(columns=sum(list(colnames.values()),[]),inplace=True)
        print("")
    if thetype == "synthetic":
        M = 3
        Tmax = 1000
        Dmax = 2
        N = 4

        d_out = 2
        #test()
        names = ["m{}".format(i+1) for i in range(M)]

        # Create signals from M modalities, all with the same dimension and length, with irregular sampling
        D = torch.ones(M).long() * Dmax
        T = [Tmax] + torch.randint(3, Tmax, (M-1,)).long().numpy().tolist()
        #T = [Tmax] + [Tmax]*(M-1)  #torch.randint(3, Tmax, (M-1,)).long().numpy().tolist()

        timelines = {k: torch.sort(torch.randn(t).abs(), descending=False).values for k,t in zip(names, T)}
        #timelines = {k: torch.arange(t, dtype=torch.float) for k,t in zip(names, T)}

        data = {k: torch.rand(N, t, d) for i,(k,t,d) in enumerate(zip(names, T, D))}

        y = compute_target(data, timelines, d_out)
    return data, timelines, y
if __name__ == "__main__":

    device = "cpu"
    if torch.cuda.is_available():
        torch.cuda.init()
        device = "cuda:0"
    
    data, timelines, y = get_data("robot")

    model =  CAMD(M, Dmax, Dmax, Dmax, n_layers=1, activation="relu", layernorm=False, skipconnections=True, skiptemperature=True).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    L = []

    X = prep_data(data, timelines, device=device)
    
    num_epochs = 1000
    every_e = num_epochs // 20
    figsize = (15, 5)
    if device=="cpu":
        fig, ax = plt.subplots(figsize=figsize)

    for epoch in range(num_epochs):
        yhat = model(X)

        loss = torch.nn.functional.mse_loss(yhat, y.to(device))
        loss.backward()

        optimizer.step()
        
        L.append(loss.item())
        if (epoch % every_e) == 0:
            print(epoch, L[-1])
            if device=="cpu":
                ax.cla()
                _, images = plot_data(data, timelines,target=y[0],prediction=yhat[0].detach(),dim=0,figsize=figsize,masks=False,ax=ax)
                plt.pause(0.5)
    print("")
