import torch
#import fast_transformers
import sys
from src.causal_product import causal_dot_product,causal_dot_product_ref
#from fast_transformers.causal_product import  causal_dot_product as causal_dot_product_reference

def test():
    M, Dmax, Tmax = 3, 20, 10
    N = 1
    names = ["m{}".format(i+1) for i in range(M)]
    
    easy = True

    regular = easy
    same_size = easy
    
    D = torch.ones(M).long() * Dmax
    T = [Tmax] + torch.randint(Tmax//3,Tmax,(M-1,)).long().numpy().tolist()# [Tmax]*()
    if same_size:
        T = [Tmax] + [Tmax]*(M-1)
    
    timelines = {k: torch.sort(torch.randn(t).abs(),descending=False).values*100 for k,t in zip(names, T)}
    if regular:
        timelines = {k: torch.arange(t, dtype=torch.float)+10 for k,t in zip(names, T)}

    data = {k: torch.randn(N, t, d) for k,t,d in zip(names, T, D)}

    x1 = data["m1"]
    t1 = timelines["m1"].reshape(-1)

    x2 = data["m2"]
    t2 = timelines["m2"].reshape(-1)
    
    Wcpu = torch.randn((Dmax, Dmax))
    Wgpu = Wcpu.clone()  #
    Wcpuref = Wcpu.clone()  #
    Wgpuref = Wcpu.clone()  #

    Wcpuref.requires_grad_(True)
    Wgpu.requires_grad_(True)
    Wcpu.requires_grad_(True)
    Wgpuref.requires_grad_(True)


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
    print("\n".join([str(d.shape) for d in [queries@Wgpu, keys@Wgpu, values, t1, t2]]))

    print("Tq:", t1)
    print("Tkv:", t2)
    print(t1.reshape(-1,1) >= t2.reshape(1,-1))
    
    if regular & same_size:
        outputCPUREF = causal_dot_product_ref((queries@Wcpuref), (keys@Wcpuref), values, t1, t2)
        lossCPUREF = (outputCPUREF**2).sum()
        lossCPUREF.backward()
    
    
    outputCPU = causal_dot_product((queries@Wcpu), (keys@Wcpu), values, t1, t2)
    lossCPU = (outputCPU**2).sum()
    lossCPU.backward()

    if regular & same_size:
        print("================")
        print("mean|CPU-REF|^2=", (outputCPU.detach()-outputCPUREF.detach()).square().mean())
        print("mean|GradCPU-GradCPUREF|^2=", (Wcpu.grad-Wcpuref.grad).square().mean())

    if torch.cuda.is_available():
        outputGPU = causal_dot_product((queries@Wgpu).cuda(), (keys@Wgpu).cuda(), values.cuda(), t1.cuda(), t2.cuda())

        lossGPU = (outputGPU**2).sum()
        lossGPU.backward()

        if regular & same_size:
            outputGPUREF = causal_dot_product_ref((queries@Wgpuref).cuda(), (keys@Wgpuref).cuda(), values.cuda(), t1.cuda(), t2.cuda())
        
            lossGPUREF = (outputGPUREF**2).sum()
            lossGPUREF.backward()
            print("================")
            print("mean|CPUREF-GPUREF|^2=", (outputCPUREF.detach()-outputGPUREF.detach().cpu()).square().mean())
            print("mean|GradCPUREF-GradGPUREF|^2=", (Wcpuref.grad-Wgpuref.grad.cpu()).square().mean())

            print("================")
            print("mean|GPU-GPUREF|^2=", (outputGPU.detach()-outputGPUREF.detach()).square().mean())
            print("mean|GradGPU-GradGPUREF|^2=", (Wgpu.grad.cpu()-Wgpuref.grad).square().mean())

        print("================")
        print("mean|GPU-CPU|^2=", (outputGPU.detach().cpu()-outputCPU.detach()).square().mean())
        print("mean|GradGPU-GradCPU|^2=", (Wgpu.grad.cpu()-Wcpu.grad).square().mean())


    sys.exit(0)