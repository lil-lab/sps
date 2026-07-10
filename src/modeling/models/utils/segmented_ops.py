"""
Document-aware token counting used for diagnostic stats.
"""

from __future__ import annotations

import torch
from torch import Tensor


def masked_per_document_count(
    documents_idx_BxT: Tensor,
    mask_BxT: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Compute token counts per document after excluding masked-out positions.
    """
    B, T = documents_idx_BxT.shape
    device = documents_idx_BxT.device
    mask_BxT = mask_BxT.to(dtype=torch.bool, device=device)

    max_doc_id = T
    batch_offset = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1) * max_doc_id
    flat_doc_idx = (batch_offset + documents_idx_BxT).reshape(-1)

    num_slots = B * max_doc_id
    doc_count = torch.zeros(num_slots, device=device, dtype=torch.long)
    doc_count.scatter_add_(0, flat_doc_idx, mask_BxT.reshape(-1).to(torch.long))
    doc_mask = doc_count > 0
    return doc_count, doc_mask
