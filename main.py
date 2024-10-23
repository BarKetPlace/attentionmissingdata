import torch
import os, sys

#os.environ['CUDA_LAUNCH_BLOCKING']="1"
#os.environ['TORCH_USE_CUDA_DSA'] = "1"
#import fast_transformers
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

#from fast_transformers.attention import LinearAttention, CausalLinearAttention

elu_feature_map = lambda x: torch.nn.functional.elu(x) + 1


from src.dataset import compute_target, prep_data, plot_data, compute_target4 as compute_target
from src.model import CAMD
from test import test
import pandas as pd
import numpy as np

def encode_time(X, device="cpu",tlim=None):
    timelines = {}
    for k,v in X.items():
        colnames = v.columns.tolist()
        v["deltas"] = np.diff(v.index.values,prepend=0)
        v["timeline"] = v.index
        timelines[k] = torch.from_numpy(v["timeline"].copy().values)[:tlim]
        v = v.reset_index(drop=True)
        X[k] = torch.from_numpy(v[colnames+["deltas","timeline"]].values.astype(np.float32)[:tlim]).to(device=device).unsqueeze(0).unsqueeze(0)
    return X, timelines

def corrupt(X, p=0):
    """X: a dictionary of modalities {modality_name: pd.DataFrame, ...}
        $p in [0,1]$: the percentage of missing data in the output.
        """
    assert(p < 1)
    if p == 0:
        return X
        
    for k, v in X.items():
        if k != "reference":
            idx_keep = np.zeros(v.shape[0], dtype=bool)
            idx_keep[np.random.permutation(v.shape[0])[int(v.shape[0]*p):]] =True
            X[k] = v[idx_keep].copy()
    return X

def get_data(thetype="synthetic",device="cpu",tlim=None):
    if thetype == "robot":
        # https://archive.ics.uci.edu/dataset/963/ur3+cobotops
        fname = "data/dataset_02052023.xlsx"
        df = pd.read_excel(fname).drop(columns=["Num"])
        df.columns=[s.strip() for s in df.columns]
        df["cycle"]/=df["cycle"].max()

        for stem in ["Temperature"]:
            all_cols = [s for s in df.columns if s.startswith(stem)]
            themin=df[all_cols].min().min()
            themax=df[all_cols].max().max()

            df[all_cols] = (df[all_cols]-themin)/(themax-themin)

        df["Timestamp"] = pd.to_datetime(df["Timestamp"].apply(lambda s: s.replace("\"","")),format="%Y-%m-%dT%H:%M:%S.%f%z")
        timeline = (df["Timestamp"]-df["Timestamp"].iloc[0]).dt.total_seconds().values
        df.index = timeline
        df.drop(columns=["Timestamp"],inplace=True)
        M = 6
        reference_cols = ["Tool_current", "cycle"]
        
        target_cols = ['Robot_ProtectiveStop', 'grip_lost']

        colnames = {**{"reference":reference_cols},**{"m{}".format(m):[s for s in df.columns if s.endswith(str(m))] for m in range(M)}}
        X = {m: df[cols].copy() for m,cols in colnames.items()}
        df.drop(columns=sum(list(colnames.values()),[]),inplace=True)
        
        df[target_cols[0]] = df[target_cols[0]].fillna(0)
        y = torch.from_numpy(df[target_cols].values.astype(np.float32)[:tlim]).unsqueeze(0)
        X = corrupt(X, p=0.5)
        X, timelines = encode_time(X, device=device,tlim=tlim)
        assert(y.shape[1] == X["reference"].shape[2])
        #plt.close(); plt.plot(df[targets[1]].values); plt.show()

        print("")


    if thetype == "synthetic":
        M = 3
        Tmax = 1000
        Dmax = 2
        N = 4

        d_out = 2
        
        names = ["reference"] + ["m{}".format(i+1) for i in range(1,M)]

        # Create signals from M modalities, all with the same dimension and length, with irregular sampling
        D = torch.ones(M).long() * Dmax
        T = [Tmax] + torch.randint(3, Tmax, (M-1,)).long().numpy().tolist()
        #T = [Tmax] + [Tmax]*(M-1)  #torch.randint(3, Tmax, (M-1,)).long().numpy().tolist()

        timelines = {k: torch.sort(torch.randn(t).abs(), descending=False).values for k,t in zip(names, T)}
        #timelines = {k: torch.arange(t, dtype=torch.float) for k,t in zip(names, T)}

        data = {k: torch.rand(N, t, d) for i,(k,t,d) in enumerate(zip(names, T, D))}

        y = compute_target(data, timelines, d_out)
        X = prep_data(data, timelines, device=device)
    return X, timelines, y

if __name__ == "__main__":

    device = "cpu"
    if torch.cuda.is_available():
        torch.cuda.init()
        device = "cuda:0"
    plot=False

    X, timelines, y = get_data("robot",device=device, tlim=1000)
    modality_dimensions = {k: v.shape[-1] for k,v in X.items()}
    d_out = y.shape[2]
    d_qk = 6
    M = len(X)
    model =  CAMD(modality_dimensions, d_out, d_qk, n_layers=3, activation="relu", layernorm=True, skipconnections=True, skiptemperature=True).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    L = []
    
    num_epochs = 1000
    every_e = num_epochs // 20
    figsize = (15, 5)
    if plot and (device == "cpu"):
        fig, ax = plt.subplots(figsize=figsize)

    for epoch in range(num_epochs):
        yhat = model(X)

        loss = torch.nn.functional.mse_loss(yhat, y.to(device))
        loss.backward()

        optimizer.step()
        
        L.append(loss.item())
        if (epoch % every_e) == 0:
            print(epoch, L[-1])
            if plot and (device=="cpu"):
                ax.cla()
                _, images = plot_data(data, timelines,target=y[0],prediction=yhat[0].detach(),dim=0,figsize=figsize,masks=False,ax=ax)
                plt.pause(0.5)
    print("")
