import torch
import numpy as np

def get_ray_maps(intrinsic, c2w, H, W):
    ###
    ### intrinsic: vt, 3, 3
    ### c2w:       vt, 4, 4
    ### rays:      vt, H, W, 3 and vt, H, W, 3
    ### 
    vt = intrinsic.shape[0]
    fx, fy, cx, cy = intrinsic[:,0,0].unsqueeze(1).unsqueeze(2), intrinsic[:,1,1].unsqueeze(1).unsqueeze(2), intrinsic[:,0,2].unsqueeze(1).unsqueeze(2), intrinsic[:,1,2].unsqueeze(1).unsqueeze(2)
    i, j = torch.meshgrid(torch.linspace(0.5, W-0.5, W, device=c2w.device), torch.linspace(0.5, H-0.5, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    i = i.unsqueeze(0).repeat(vt,1,1)
    j = j.unsqueeze(0).repeat(vt,1,1)
    dirs = torch.stack([(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:,np.newaxis,np.newaxis, :3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_o = c2w[:, :3,-1].unsqueeze(1).unsqueeze(2).repeat(1,H,W,1)
    viewdir = rays_d/torch.norm(rays_d, dim=-1, keepdim=True)
    return rays_o, viewdir