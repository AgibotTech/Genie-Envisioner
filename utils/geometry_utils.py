import torch
import torch.nn.functional as F
from einops import rearrange


def resize_traj_and_ray(traj_n_ray, mem_size, future_size, height, width):
    '''
    traj_n_ray: bv c t h w
    '''
    orig_t = traj_n_ray.shape[3]
    try:
        assert orig_t > (mem_size + future_size)
    except:
        breakpoint()
        
    n_view = traj_n_ray.shape[2]

    mem = traj_n_ray[:, :, :mem_size]
    mem = rearrange(mem, 'bv c t h w -> (bv t) c h w')
    mem = F.interpolate(mem, (height, width), mode='bilinear')
    mem = rearrange(mem, '(bv t) c h w -> bv c t h w', t=mem_size)

    future = traj_n_ray[:, :, mem_size:]  # bv c t h w
    future = F.interpolate(future, (future_size, height, width), mode='trilinear')

    out = torch.cat([mem, future], dim=2)
    return out
