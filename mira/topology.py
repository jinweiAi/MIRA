"""Causal two-stage topology for one unpadded multimodal sequence."""
import torch

def build_access_mask(groups, stage=1, device=None):
    """P: prompt text; V: image; E: parallel latent; R: assistant header; A: answer.

    Stage I removes direct V -> P/R/A information flow. E queries read P/V
    and their own diagonal, never other E slots. Stage II reopens the native
    causal visual path for P/R/A. All stages preserve physical causal order.
    """
    if isinstance(groups, str) or not groups or any(g not in "PVERA" for g in groups):
        raise ValueError("groups must be a nonempty sequence of P/V/E/R/A")
    if isinstance(stage, bool) or stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    codes = torch.tensor(["PVERA".index(g) for g in groups], device="cpu")
    rules = torch.zeros(5, 5, dtype=torch.bool)
    rules[0, 0] = True
    rules[0, 1] = stage == 2
    rules[1, :2] = True
    rules[2, :2] = True
    rules[3:, :] = True
    rules[3:, 1] = stage == 2
    allowed = rules[codes[:, None], codes[None, :]]
    allowed.fill_diagonal_(True)
    return allowed.tril().to(device=device)
