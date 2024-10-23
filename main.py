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
        timeline = (df["Timestamp"]-df["Timestamp"].iloc[0]).dt.total_seconds().values/60
        df.index = timeline
        df.drop(columns=["Timestamp"],inplace=True)
        M = 6
        reference_cols = ["Tool_current", "cycle"]
        df = df[(df[reference_cols].notna().sum(1)==2).values].copy()
        target_cols = ['Robot_ProtectiveStop', 'grip_lost']

        colnames = {**{"reference":reference_cols},**{"m{}".format(m):[s for s in df.columns if s.endswith(str(m))] for m in range(M)}}
        X = {m: df[cols].dropna().copy() for m,cols in colnames.items()}
        df.drop(columns=sum(list(colnames.values()),[]),inplace=True)
        
        df[target_cols[0]] = df[target_cols[0]].fillna(0)
        ydata = torch.from_numpy(df[target_cols].values.astype(np.float32)[:tlim]).unsqueeze(0)
        y=[ydata,target_cols]
        X = corrupt(X, p=0.5)
        X, timelines = encode_time(X, device=device,tlim=tlim)
        assert(ydata.shape[1] == X["reference"].shape[2])
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

        y = [compute_target(data, timelines, d_out), None]
        X = prep_data(data, timelines, device=device)
    return X, timelines, y

def compute_scores(yhat,y,names):
    """y and yhat of size (N,T,q)
    names a list of length q (the number of targets)"""
    # Accuracy AUC  Kappa  Precision Recall F1-Score
    out = {}
    for i,thename in enumerate(names):
        outtmp = dict(
            binary_auroc=binary_auroc(yhat[...,i],y[...,i].int()),
            binary_auprc=binary_auprc(yhat[...,i],y[...,i].int()),
            binary_precision=binary_precision(yhat[...,i],y[...,i].int()),
            binary_recall=binary_recall(yhat[...,i],y[...,i].int()),
            binary_f1_score=binary_f1_score(yhat[...,i],y[...,i]),
            binary_accuracy=binary_accuracy(yhat[...,i],y[...,i]))
        out = {**out, **{thename+"_"+k: v for k,v in outtmp.items()}}
    return out
    
if __name__ == "__main__":

    device = "cpu"
    if torch.cuda.is_available():
        torch.cuda.init()
        device = "cuda:0"
    plot=not (device=="cuda:0")

    X, timelines, yout = get_data("robot",device=device, tlim=100)
    
    y, target_cols = yout

    modality_dimensions = {k: v.shape[-1] for k,v in X.items()}
    d_out = y.shape[2]
    d_qk = 6
    M = len(X)
    layer_opts=dict(layernorm=False, skipconnections=False, skiptemperature=False)
    model =  CAMD(modality_dimensions, d_out, d_qk, n_layers=3, activation="relu", **layer_opts).to(device)
    def init_weights_to_zero(model):
        for param in model.parameters():
            param.data.zero_()  # Set the data of each parameter to 0

    # Initialize the model's parameters to 0
    init_weights_to_zero(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    from torcheval.metrics.functional import r2_score, binary_auprc, binary_auroc, binary_precision, binary_recall, binary_f1_score, binary_accuracy
    from torchmetrics.functional.regression.spearman import _spearman_corrcoef_compute
    L = []

    num_epochs = 10000
    every_e = num_epochs // 20
    figsize = (15, 5)
    if plot and (device == "cpu"):
        fig, ax = plt.subplots(figsize=figsize)

    for epoch in range(num_epochs):
        yhat = torch.nn.functional.sigmoid(model(X))
        loss0 = torch.nn.functional.cross_entropy(yhat[0,:,0], y[0,:,0].to(yhat.device))
        loss1 = torch.nn.functional.cross_entropy(yhat[0,:,1], y[0,:,1].to(yhat.device))

        loss = (loss0+loss1)/2 #
        #loss = torch.nn.functional.mse_loss(yhat, y.to(yhat.device))
        loss.backward()
        
        optimizer.step()
        #scores = compute_scores(yhat[0].detach(), y[0].detach().to(yhat.device), target_cols)
        L.append(loss.item())
        if (epoch % every_e) == 0:
            print(epoch, L[-1])
            if plot and (device=="cpu"):
                ax.cla()
                ax.plot(timelines["reference"],y[0,:,0], color="darkred", label="True "+target_cols[0])
                ax.plot(timelines["reference"],yhat[0,:,0].detach(), color="black", label="Predicted "+target_cols[0])
                ax.plot(timelines["reference"],y[0,:,1], color="darkred", label="True "+target_cols[1],linestyle="--")
                ax.plot(timelines["reference"],yhat[0,:,1].detach(), color="black", label="Predicted "+target_cols[1],linestyle="--")
                ax.legend()
                #_, images = plot_data(X, timelines, target=y[0],prediction=yhat[0].detach(),dim=0,figsize=figsize,masks=False,ax=ax)
                plt.pause(0.5)
    scores = compute_scores(yhat[0].detach(), y[0].detach().to(yhat.device), target_cols)
    print(scores)
    print("")
