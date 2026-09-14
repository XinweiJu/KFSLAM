import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreNet(nn.Module):
    """
    Simple MLP ScoreNet.
    Input: pair_feats (B,T,F)
    Output: scores (B,T)
    """
    def __init__(self, in_features=8, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, pair_feats):
        if pair_feats.dim() == 5:
            # global average pool if spatial
            B, T, C, H, W = pair_feats.shape
            pair_feats = pair_feats.view(B * T, C, H, W).mean(dim=[2, 3])
        else:
            B, T, F = pair_feats.shape
            pair_feats = pair_feats.view(B * T, F)
        s = self.mlp(pair_feats)
        return s.view(-1, T)


class GumbelTopKSelector(nn.Module):
    """
    Gumbel-Top-k selector with straight-through gradients.
    Returns (weights, topk_indices).
    """
    def __init__(self, k=3, temperature=1.0, stochastic=True, normalize_to_k=True):
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.stochastic = stochastic
        self.normalize_to_k = normalize_to_k

    @staticmethod
    def _sample_gumbel(shape, device, dtype):
        u = torch.rand(shape, device=device, dtype=dtype).clamp_(1e-6, 1 - 1e-6)
        return -torch.log(-torch.log(u))

    def forward(self, logits):
        B, T = logits.shape
        k_eff = min(self.k, T)
        if self.training and self.stochastic:
            g = self._sample_gumbel((B, T), logits.device, logits.dtype)
            perturbed = (logits + g) / max(self.temperature, 1e-6)
        else:
            perturbed = logits / max(self.temperature, 1e-6)

        topk_val, topk_idx = torch.topk(perturbed, k=k_eff, dim=1)
        hard = torch.zeros_like(logits)
        hard.scatter_(1, topk_idx, 1.0)
        hard = hard / (hard.sum(dim=1, keepdim=True) + 1e-8)

        soft = F.softmax(perturbed, dim=1)
        if self.normalize_to_k:
            soft = soft * (k_eff / (soft.sum(dim=1, keepdim=True) + 1e-8))

        weights = hard + soft - soft.detach()
        return weights, topk_idx
