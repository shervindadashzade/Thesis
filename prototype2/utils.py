import torch

def mlm_mask_torch(
    input_ids: torch.Tensor,        # (L,)
    vocab_size: int,
    mask_prob: float = 0.15,
    mask_token_id: int = 1,
    special_token_ids: set = {0, 1},
    generator: torch.Generator | None = None,
):
    assert input_ids.dim() == 1, "expects a 1D sequence"
    device = input_ids.device
    seq = input_ids.clone()
    labels = torch.full_like(seq, fill_value=-100)

    special_mask = torch.zeros_like(seq, dtype=torch.bool)
    for s in special_token_ids:
        special_mask |= (seq == s)
    candidates = (~special_mask).nonzero(as_tuple=False).squeeze(-1)

    n_to_mask = max(1, int(round(mask_prob * candidates.numel()))) if candidates.numel() else 0
    if n_to_mask == 0:
        return seq, labels, torch.empty(0, dtype=torch.long, device=device)

    perm = torch.randperm(candidates.numel(), generator=generator, device=device)
    mask_idx = candidates[perm[:n_to_mask]]
    labels[mask_idx] = seq[mask_idx]

    n80 = int(round(0.8 * n_to_mask))
    n10 = int(round(0.1 * n_to_mask))
    idx80 = mask_idx[:n80]
    idx10_rand = mask_idx[n80:n80+n10]
    idx10_keep = mask_idx[n80+n10:]

    # 80% → MSK
    seq[idx80] = mask_token_id

    # 10% → random token in [2, vocab_size-1], not equal to original
    low = 2
    if vocab_size <= low:
        raise ValueError("vocab_size must be > 2 to sample random tokens.")
    # sample; fix collisions
    rand_tokens = torch.randint(low, vocab_size, (idx10_rand.numel(),), generator=generator, device=device, dtype=seq.dtype)
    collisions = rand_tokens == input_ids[idx10_rand]
    while collisions.any():
        rand_tokens[collisions] = torch.randint(low, vocab_size, (collisions.sum().item(),), generator=generator, device=device, dtype=seq.dtype)
        collisions = rand_tokens == input_ids[idx10_rand]
    seq[idx10_rand] = rand_tokens
    # 10% keep → no change

    return seq, labels, mask_idx