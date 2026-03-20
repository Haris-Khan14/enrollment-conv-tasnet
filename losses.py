import torch
import torch.nn.functional as F

def si_snr(est, target, eps=1e-8):
    """
    est:    [B, 1, T]
    target: [B, 1, T]
    """

    est = est.squeeze(1)
    target = target.squeeze(1)

    target = target - torch.mean(target, dim=1, keepdim=True)
    est = est - torch.mean(est, dim=1, keepdim=True)

    s_target = torch.sum(est * target, dim=1, keepdim=True) \
               * target / (torch.sum(target**2, dim=1, keepdim=True) + eps)

    e_noise = est - s_target

    si_snr_val = 10 * torch.log10(
        (torch.sum(s_target**2, dim=1) + eps) /
        (torch.sum(e_noise**2, dim=1) + eps)
    )

    return -torch.mean(si_snr_val)