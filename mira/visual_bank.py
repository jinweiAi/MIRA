"""Frozen Qwen2.5-VL visual-bank extractor for transformers==4.51.3.

Uses public module forward hooks for inspecting selected visual blocks.
No global buffer survives the function. Do not run concurrent calls on one
vision module while these hooks are installed. Production should use an explicit
forward_with_bank method following this same ordering and verify equivalence.
"""
import torch


def _merged_visual_output(output):
    """Return the merger output across supported Transformers return types.

    Transformers 4.51 returns the merged visual embeddings directly, while
    newer releases wrap them in ``BaseModelOutputWithPooling`` as
    ``pooler_output``.  The latter is the value that corresponds to Qwen's
    language-side image embeddings; ``last_hidden_state`` is still the
    pre-merger patch sequence and must not be used here.
    """
    if torch.is_tensor(output):
        return output
    pooled = getattr(output, "pooler_output", None)
    if torch.is_tensor(pooled):
        return pooled
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    raise TypeError(
        "Qwen vision forward must return a Tensor or an output with "
        "tensor pooler_output"
    )


@torch.no_grad()
def extract_visual_bank(vision, pixel_values, grid_thw, layer_indices=tuple(range(32))):
    if not layer_indices or len(set(layer_indices)) != len(layer_indices):
        raise ValueError("candidate indices must be unique and nonempty")
    if min(layer_indices) < 0 or max(layer_indices) >= len(vision.blocks):
        raise ValueError("candidate block index out of range")
    captured = {}
    handles = []
    try:
        for index in layer_indices:
            def capture(module, args, output, index=index):
                captured[index] = output
            handles.append(vision.blocks[index].register_forward_hook(capture))
        original = _merged_visual_output(vision(pixel_values, grid_thw=grid_thw))
    finally:
        for handle in handles:
            handle.remove()
    window_index, _ = vision.get_window_index(grid_thw)
    reverse = window_index.argsort()
    projected = []
    for index in layer_indices:
        # Within each merger group, four patch rows remain consecutive.
        # Merge first, then restore group order. Do not reverse raw patch rows.
        merged = vision.merger(captured[index])
        # Transformers may return a CPU window index even when the visual
        # tower runs on CUDA.  Advanced indexing must use the merger output's
        # device explicitly.
        projected.append(merged[reverse.to(device=merged.device)])
    bank = torch.stack(projected, dim=1)  # [sum_images N_image, M, d]
    lengths = grid_thw.prod(-1) // (vision.spatial_merge_size ** 2)
    if bank.shape[0] != int(lengths.sum()):
        raise AssertionError("merger/token-count mismatch")
    if len(vision.blocks) - 1 in layer_indices:
        final_index = layer_indices.index(len(vision.blocks) - 1)
        torch.testing.assert_close(bank[:, final_index], original)
    return original, bank, lengths
