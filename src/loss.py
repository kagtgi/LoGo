import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

# ==============================================================#
# Core Loss Functions
# ==============================================================#

def dice_loss(pred, target, eps=1e-5):
    """
    pred: logits or single-channel prediction (B,1,H,W) or (B,H,W)
    target: same shape, float {0,1}
    """
    #pred = torch.sigmoid(pred)
    # flatten over batch+spatial
    num = 2 * (pred * target).sum()
    den = pred.sum() + target.sum() + eps
    return 1 - num / den


def focal_tversky_loss(pred, target, alpha=0.3, beta=0.7, gamma=0.75, eps=1e-6):
    #pred = torch.sigmoid(pred)
    target = target.float()
    tp = (pred * target).sum()
    fp = ((1 - target) * pred).sum()
    fn = (target * (1 - pred)).sum()
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1 - tversky) ** gamma


def boundary_loss(pred, target):
    """
    Improved boundary loss: normalized signed distance weighting
    focusing on pixels near edges.
    pred: logits or single channel
    target: binary mask (B,1,H,W)
    """
    #pred = torch.sigmoid(pred)
    target = target.float()
    target_np = target.detach().cpu().numpy()

    sdf = np.zeros_like(target_np, dtype=np.float32)
    for b in range(target_np.shape[0]):
        # assume channel-first mask with channel=1
        mask = target_np[b, 0].astype(bool)
        inv_mask = ~mask

        # Edge cases: all zeros or all ones
        if not mask.any():
            sdf[b, 0] = distance_transform_edt(inv_mask)
        elif not inv_mask.any():
            sdf[b, 0] = -distance_transform_edt(mask)
        else:
            sdf[b, 0] = distance_transform_edt(inv_mask) - distance_transform_edt(mask)

    sdf = torch.from_numpy(sdf).to(pred.device).float()
    max_abs_sdf = torch.max(torch.abs(sdf))
    if max_abs_sdf > 1e-6:
        sdf = sdf / max_abs_sdf  # Normalize for stability

    return (torch.abs(pred - target) * torch.abs(sdf)).mean()


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Standard focal BCE on probabilities.
    pred: logits or single-channel; we apply sigmoid inside.
    """
    #pred = torch.sigmoid(pred)
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.where(target == 1, pred, 1 - pred)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * bce).mean()


def diversity_loss(proto_list, temperature=1.0):
    """
    Encourages prototype diversity for list-based prototypes.

    Accepts:
      - proto_list: List of tuples [(fg_proto, bg_proto), ...] where each is [K, D]
                    OR a single tensor (K, D) or (B, K, D) for backward compatibility
    Returns scalar tensor (averaged across batch).
    """
    # Handle list of tuples format
    if isinstance(proto_list, list):
        if len(proto_list) == 0:
            return torch.tensor(0.0, device='cpu', requires_grad=True)

        losses = []
        for fg_proto, bg_proto in proto_list:
            # Compute diversity for FG prototypes
            if fg_proto is not None and fg_proto.numel() > 0:
                fg_loss = _diversity_single(fg_proto, temperature)
                losses.append(fg_loss)
            # Compute diversity for BG prototypes
            if bg_proto is not None and bg_proto.numel() > 0:
                bg_loss = _diversity_single(bg_proto, temperature)
                losses.append(bg_loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device='cpu', requires_grad=True)
        return torch.stack(losses).mean()

    # Backward compatibility: handle tensor formats
    return _diversity_single(proto_list, temperature)


def _diversity_single(prototypes, temperature=1.0):
    """Helper function for single prototype tensor."""
    if prototypes is None:
        return torch.tensor(0.0, device='cpu', requires_grad=True)

    prototypes = torch.as_tensor(prototypes)

    # Single set (K, D)
    if prototypes.dim() == 2:
        K, D = prototypes.shape
        if K <= 1:
            return torch.tensor(0.0, device=prototypes.device, requires_grad=True)
        pnorm = F.normalize(prototypes, p=2, dim=1)
        sim = torch.matmul(pnorm, pnorm.t()) / float(temperature)
        mask = ~torch.eye(K, dtype=torch.bool, device=prototypes.device)
        off_diag = sim[mask]
        return (off_diag ** 2).mean()

    # Batched (B, K, D)
    elif prototypes.dim() == 3:
        B, K, D = prototypes.shape
        if K <= 1:
            return torch.tensor(0.0, device=prototypes.device, requires_grad=True)
        pnorm = F.normalize(prototypes, p=2, dim=2)
        sim = torch.matmul(pnorm, pnorm.transpose(1, 2)) / float(temperature)
        mask = ~torch.eye(K, dtype=torch.bool, device=prototypes.device).unsqueeze(0)
        off_diag = sim[mask.expand_as(sim)].view(B, -1)
        per_batch = (off_diag ** 2).mean(dim=1)
        return per_batch.mean()
    else:
        raise ValueError("prototypes must be 2D (K,D) or 3D (B,K,D)")


def proto_orthogonal_loss(fg_proto, bg_proto):
    """
    Regularizes foreground and background prototypes to be orthogonal.

    Accepts:
      - fg_proto: List of tuples format OR (K_fg, D) or (B, K_fg, D)
      - bg_proto: List of tuples format OR (K_bg, D) or (B, K_bg, D)

    Returns scalar tensor.
    """
    # Handle list of tuples format (new format)
    if isinstance(fg_proto, list) and isinstance(bg_proto, list):
        if len(fg_proto) == 0 or len(bg_proto) == 0:
            return torch.tensor(0.0, device='cpu', requires_grad=True)

        # fg_proto and bg_proto should be the same list
        losses = []
        for (fg, bg) in zip(fg_proto, bg_proto):
            if fg is None or bg is None:
                continue
            fg_t, bg_t = (fg[0], fg[1]) if isinstance(fg, tuple) else (fg, bg)
            loss = _proto_orthogonal_single(fg_t, bg_t)
            losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device='cpu', requires_grad=True)
        return torch.stack(losses).mean()

    # For refined_proto_list format: list of tuples
    if isinstance(fg_proto, list) and not isinstance(bg_proto, list):
        # fg_proto is list of tuples, extract fg and bg from it
        losses = []
        for fg_t, bg_t in fg_proto:
            loss = _proto_orthogonal_single(fg_t, bg_t)
            losses.append(loss)
        if len(losses) == 0:
            return torch.tensor(0.0, device='cpu', requires_grad=True)
        return torch.stack(losses).mean()

    # Backward compatibility: handle tensor formats
    return _proto_orthogonal_single(fg_proto, bg_proto)


def _proto_orthogonal_single(fg_proto, bg_proto):
    """Helper function for single prototype pair."""
    if fg_proto is None or bg_proto is None:
        return torch.tensor(0.0, device='cpu', requires_grad=True)

    fg = torch.as_tensor(fg_proto)
    bg = torch.as_tensor(bg_proto)

    # If both are 2D
    if fg.dim() == 2 and bg.dim() == 2:
        if fg.shape[0] == 0 or bg.shape[0] == 0:
            return torch.tensor(0.0, device=fg.device, requires_grad=True)
        fg_n = F.normalize(fg, p=2, dim=1)
        bg_n = F.normalize(bg, p=2, dim=1)
        sim = torch.matmul(fg_n, bg_n.t())
        return sim.abs().mean()

    # If batched
    elif fg.dim() == 3 and bg.dim() == 3:
        if fg.shape[1] == 0 or bg.shape[1] == 0:
            return torch.tensor(0.0, device=fg.device, requires_grad=True)
        fg_n = F.normalize(fg, p=2, dim=2)
        bg_n = F.normalize(bg, p=2, dim=2)
        sim = torch.matmul(fg_n, bg_n.transpose(1, 2))
        return sim.abs().mean()

    # Mixed shapes
    elif fg.dim() == 3 and bg.dim() == 2:
        bg = bg.unsqueeze(0).expand(fg.shape[0], -1, -1)
        return _proto_orthogonal_single(fg, bg)
    elif fg.dim() == 2 and bg.dim() == 3:
        fg = fg.unsqueeze(0).expand(bg.shape[0], -1, -1)
        return _proto_orthogonal_single(fg, bg)
    else:
        raise ValueError("Unsupported prototype tensor shapes")


def small_organ_loss(pred, target, lambda_ftl=1.0, lambda_boundary=0.5):
    loss_ftl = focal_tversky_loss(pred, target)
    loss_bnd = boundary_loss(pred, target)
    return lambda_ftl * loss_ftl + lambda_boundary * loss_bnd


def weighted_cross_entropy_loss(pred, target, class_weights=None, ignore_index=-100):
    """
    Weighted Cross-Entropy Loss for segmentation with class imbalance.

    Args:
        pred: logits with shape (B, C, H, W) where C is number of classes
        target: ground truth with shape (B, H, W) or (B, 1, H, W) containing class indices
        class_weights: tensor of shape (C,) with weights for each class
                      e.g., torch.FloatTensor([0.05, 1.0]) gives background weight=0.05, foreground=1.0
        ignore_index: label index to ignore in loss calculation

    Returns:
        scalar loss tensor
    """
    # Handle target shape - ensure it's (B, H, W)
    if target.dim() == 4:
        target = target.squeeze(1)  # (B, 1, H, W) -> (B, H, W)

    # Convert float target to long for CrossEntropyLoss
    target = target.long()

    # Default weights if not provided
    if class_weights is None:
        class_weights = torch.FloatTensor([0.05, 1.0]).to(pred.device)
    elif not isinstance(class_weights, torch.Tensor):
        class_weights = torch.FloatTensor(class_weights).to(pred.device)
    else:
        class_weights = class_weights.to(pred.device)

    # Create criterion
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=ignore_index,
        reduction='mean'
    )

    return criterion(pred, target)


# ==============================================================#
# Combined SOTA Segmentation Loss
# ==============================================================#

def sota_segmentation_loss(pred, s_pred, target, s_target,
                           fg_proto, bg_proto, triplet_loss,
                           lambda_dice=1.0, lambda_focal=2.0,
                           lambda_proto=0.2, lambda_div=0.2,
                           lambda_ce=0.5, class_weights=None):
    """
    Balanced loss combining Dice, Focal, Boundary, Prototype, Triplet, and Cross-Entropy components.

    pred, s_pred: model outputs (logits) with either shape (B,2,H,W) or (B,1,H,W)
    target, s_target: masks (B,1,H,W)
    fg_proto, bg_proto: prototypes - can be:
                        - List of tuples [(fg, bg), ...] (new format)
                        - Single proto_list (for refined_proto_list)
                        - Tensors (K, D) or (B, K, D) (backward compatibility)
    triplet_loss: scalar tensor
    lambda_ce: weight for cross-entropy loss component
    class_weights: weights for cross-entropy loss classes
    """
    # Handle binary or 2-channel predictions
    pred_fg = pred[:, 1:2, :, :] if pred.dim() == 4 and pred.shape[1] == 2 else pred
    s_pred_fg = s_pred[:, 1:2, :, :] if s_pred.dim() == 4 and s_pred.shape[1] == 2 else s_pred

    # Ensure masks have channel dim
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if s_target.dim() == 3:
        s_target = s_target.unsqueeze(1)

    # Segmentation components
    loss_dice = dice_loss(pred_fg, target) + dice_loss(s_pred_fg, s_target)
    loss_focal = focal_loss(pred_fg, target) + focal_loss(s_pred_fg, s_target)
    loss_smallorgan = small_organ_loss(pred_fg, target) + small_organ_loss(s_pred_fg, s_target)

    # Cross-entropy loss (only if pred has 2 channels for class logits)
    loss_ce = torch.tensor(0.0, device=pred.device)
    if pred.dim() == 4 and pred.shape[1] == 2:
        loss_ce = weighted_cross_entropy_loss(pred, target, class_weights) + \
                  weighted_cross_entropy_loss(s_pred, s_target, class_weights)

    seg_loss = lambda_dice * loss_dice + lambda_focal * loss_focal + 0.1 * loss_smallorgan + lambda_ce * loss_ce

    # Prototype diversity & orthogonality (handles list or tensor formats)
    # If fg_proto is a list of tuples, compute diversity for all prototypes
    if isinstance(fg_proto, list):
        loss_div = diversity_loss(fg_proto)
        loss_ortho = proto_orthogonal_loss(fg_proto, bg_proto)
    else:
        loss_fg_div = diversity_loss(fg_proto)
        loss_bg_div = diversity_loss(bg_proto)
        loss_div = (loss_fg_div + loss_bg_div) / 2.0
        loss_ortho = proto_orthogonal_loss(fg_proto, bg_proto)

    # Total weighted loss
    total_loss = (
        seg_loss
        + loss_div
        + triplet_loss
        + 0.1 * loss_ortho
    )

    # Build stats
    stats = {
        'seg_loss': float(seg_loss.detach().cpu().item()) if isinstance(seg_loss, torch.Tensor) else float(seg_loss),
        'dice': float(loss_dice.detach().cpu().item()) if isinstance(loss_dice, torch.Tensor) else float(loss_dice),
        'focal': float(loss_focal.detach().cpu().item()) if isinstance(loss_focal, torch.Tensor) else float(loss_focal),
        'ce': float(loss_ce.detach().cpu().item()) if isinstance(loss_ce, torch.Tensor) else float(loss_ce),
        'diversity': float(loss_div.detach().cpu().item()) if isinstance(loss_div, torch.Tensor) else float(loss_div),
        'triplet_loss': float(triplet_loss.detach().cpu().item()) if isinstance(triplet_loss, torch.Tensor) else float(triplet_loss),
        'ortho': float(loss_ortho.detach().cpu().item()) if isinstance(loss_ortho, torch.Tensor) else float(loss_ortho),
    }

    return total_loss, stats
