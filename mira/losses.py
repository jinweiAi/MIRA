"""Answer/EOS supervision; no auxiliary router or diversity objective."""
import torch
from torch import Tensor
import torch.nn.functional as F

def answer_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    """[B,T,V], [B,T]. Per-example mean over supervised shifted answer tokens."""
    shifted = labels[:, 1:]
    valid = shifted != -100
    count = valid.sum(-1)
    if (count == 0).any():
        raise ValueError("every example must contain a supervised answer token")
    work = logits[:, :-1].float()
    token_loss = F.cross_entropy(
        work.reshape(-1, work.shape[-1]), shifted.reshape(-1),
        ignore_index=-100, reduction="none").reshape_as(shifted)
    return (token_loss.sum(-1) / count).mean()
