import matplotlib.pyplot as plt
import torch

import numpy as np

def compute_target(data,timelines,d_out):
    Tmax=data["m1"].shape[1]
    N =data["m1"].shape[0]

    y = torch.zeros(N, Tmax, d_out)

    time_ref = timelines["m1"]
    for itime, t in enumerate(time_ref):
        x_max_previous = torch.zeros(N,d_out)
        for k in data.keys():
            previous_data = data[k][:,timelines[k] <= t]
            if previous_data.shape[1]>0:
                x_max_previous += data[k][:,timelines[k] <= t].max(1).values
        y[:,itime, :] = x_max_previous
    return y


def compute_target2(data,timelines,d_out):
    Tmax = data["m1"].shape[1]
    N = data["m1"].shape[0]

    y = torch.zeros(N, Tmax, d_out)
    time_ref = timelines["m1"]

    for k in data.keys():
        ikeep = torch.tensor([0])#torch.randint(timelines[k].shape[0],(1,))
        y += data[k][:, [ikeep[0]], :]

    return y

def plot_data(data, timelines, target, prediction=None, dim=0, n=0,figsize=None,masks=False,ax=None):
    images=[]
    if masks:
        for k, tl in timelines.items():
            fig, ax = plt.subplots()
            colors={True:"darkgreen",False:"darkred"}
            for iq in range(timelines["m1"].shape[0]):
                c=timelines["m1"][iq]>=tl
                ax.scatter(tl, [timelines["m1"][iq]]*len(tl), c=[colors[cc.item()] for cc in c])
            ax.plot([timelines["m1"][0],timelines["m1"][-1]],[timelines["m1"][0],timelines["m1"][-1]], color="black")
            #im=ax.imshow(timelines["m1"].view(-1,1) >= tl.view(1,-1),cmap="summer",extent=[tl[0],tl[-1],timelines["m1"][0],timelines["m1"][-1]],origin="lower",interpolation="none")
            #ax.scatter([tl[0]]*timelines["m1"].shape[0],timelines["m1"],c="k",marker="X")
            #ax.scatter(tl, [timelines["m1"][0]]*tl.shape[0],c="k")

            #plt.colorbar(im)
            ax.set_ylabel("m1")
            ax.set_xlabel(k)
            #ax.set_xticks(np.linspace(0, len(tl)-1, len(tl)))  # This automatically spaces the ticks
            #ax.set_xticklabels(tl.numpy())  # Set the corresponding t1 labels
            #ax.set_yticks(np.linspace(0, len(timelines["m1"])-1, len(timelines["m1"])))  # This automatically spaces the ticks
            #ax.set_yticklabels(timelines["m1"].numpy())  # Set the corresponding t1 labels

            #ax.set_xticks(tl)  # Set the corresponding t1 labels on the x-axis

            #ax.set_xticklabels(tl)  # Set the corresponding t1 labels on the x-axis
            #ax.set_xticks(tl)
            images.append([fig,ax])
    return_plot=None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return_plot=[fig,ax]
    for i,k in enumerate(data.keys()):
        X = data[k][n,:,dim]
        timel = timelines[k]
        ax.plot(timel, X, "-", label=k, marker='o')
    ax.plot(timelines["m1"],target[:,dim],linewidth=2,marker="o",color="black",label="Target")

    ax.plot(timelines["m1"],sum(list(data.values()))[n,:,dim],linewidth=2,marker="o",color="gray",label="Sum data")

    if not (prediction is None):
        ax.plot(timelines["m1"],prediction[:,dim],linewidth=2,marker="o",color="darkred",label="Prediction")

    ax.legend()

    return return_plot,  images


def prep_data(data,timelines,device="cpu"):
    # Compute timeseries deltas
    deltas = {k: torch.diff(t, prepend=torch.tensor([t[0]])).unsqueeze(0).view(1,t.shape[0],1) for k,t in timelines.items()}
    N=data["m1"].shape[0]
    # Concatenate data and timelines
    calX = {k: torch.cat([data[k], deltas[k].expand(N,-1,-1), timelines[k].unsqueeze(0).unsqueeze(-1).expand(N,-1,-1)], dim=2).unsqueeze(1).to(device) for k in data.keys()}
    return calX
