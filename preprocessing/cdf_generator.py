import torch
import math
import numpy as np

def gen_cdf(pi, mu, sigma, max_score, device, batch_size=5000):
    num = len(pi)
    batch_size = int(min(num, batch_size))
    ticks = torch.arange(0.5, max_score + 0.5, 1, device=device).view(1, 1, max_score)
    cdf_list = []
    for b in range(math.ceil(num / batch_size)):
        pi_gpu = torch.from_numpy(pi[b * batch_size: (b+1) * batch_size]).to(device)
        mu_gpu = torch.from_numpy(mu[b * batch_size: (b+1) * batch_size]).to(device)
        sigma_gpu = torch.from_numpy(sigma[b * batch_size: (b+1) * batch_size]).to(device)
            
        normals = torch.distributions.normal.Normal(mu_gpu.unsqueeze(-1), sigma_gpu.unsqueeze(-1))
        cdf = normals.cdf(ticks)
        cdf = (cdf * pi_gpu.unsqueeze(-1)).sum(1)
        cdf = torch.where(cdf >= 0.997, torch.ones(cdf.shape, device=device), cdf)
        cdf = cdf.clamp(0, 1)
        cdf_list.append(cdf.cpu().numpy())
        torch.cuda.empty_cache()
    
    return np.concatenate(cdf_list, 0).astype(np.float64)