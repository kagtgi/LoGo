import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .loss import sota_segmentation_loss, dice_loss

def train_one_epoch(model, loader, optimizer, device, use_sota_loss=True):
    model.train()
    total_loss = 0.0
    loss_keys = ['seg_loss', 'dice', 'focal', 'ce', 'diversity', 'triplet_loss', 'ortho']
    loss_stats = {k: 0.0 for k in loss_keys}

    for batch in tqdm(loader, desc="Training"):
        supp_img = batch["support_image"].to(device)
        supp_mask = batch["support_mask"].float().to(device)
        qry_img = batch["query_image"].to(device)
        qry_mask = batch["query_mask"].float().to(device)

        optimizer.zero_grad()

        outputs = model(supp_img, supp_mask, qry_img)

        pred = outputs['sim_map']
        s_pred = outputs['reversed_map']
        # Use refined_proto_list which is List[(fg, bg), ...]
        refined_proto_list = outputs.get('refined_proto_list', None)
        triplet_loss = outputs.get('triplet_loss', torch.tensor(0.0, device=device))

        if use_sota_loss:
            loss, batch_stats = sota_segmentation_loss(
                pred, s_pred, qry_mask, supp_mask,
                refined_proto_list, None,  # Pass proto_list as fg_proto, None as bg_proto
                triplet_loss,
                lambda_dice=0.5,
                lambda_focal=1.0,
                lambda_proto=0.1,
                lambda_div=0.15
            )
            for k, v in batch_stats.items():
                if k in loss_stats:
                    loss_stats[k] += v
        else:
            loss = dice_loss(pred, qry_mask)

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += float(loss.detach().cpu().item())

    avg_loss = total_loss / len(loader)
    if use_sota_loss:
        for k in loss_stats:
            loss_stats[k] /= len(loader)
        print(f"Loss breakdown: {loss_stats}")

    return avg_loss


def distill_one_epoch(encoder, loader, optimizer, device):
    encoder.train()
    loss_global_total = 0.0
    loss_local_total = 0.0
    
    for batch in tqdm(loader, desc="Distilling"):
        qry_img = batch["query_image"].to(device)
        if qry_img.shape[1] == 1:
            qry_img = qry_img.repeat(1, 3, 1, 1)
        elif qry_img.shape[1] != 3:
            raise ValueError(f"Unexpected channel size: {qry_img.shape}")
        
        # distill method returns (loss_global, loss_local) scalar tensors
        loss_global, loss_local = encoder.distill(qry_img, optimizer)
        
        loss_global_total += loss_global.item() if hasattr(loss_global, 'item') else loss_global
        loss_local_total += loss_local.item() if hasattr(loss_local, 'item') else loss_local

    return loss_global_total / len(loader), loss_local_total / len(loader)

def evaluate(model, loader, device, use_sota_loss=True):
    model.eval()
    total_loss = 0.0

    # Track all losses
    loss_keys = ['seg_loss', 'dice', 'focal', 'ce', 'diversity', 'triplet_loss', 'ortho']
    loss_stats = {k: 0.0 for k in loss_keys}

    total_dice = 0.0
    num_batches = len(loader)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            supp_img = batch["support_image"].to(device)
            supp_mask = batch["support_mask"].float().to(device)
            qry_img = batch["query_image"].to(device)
            qry_mask = batch["query_mask"].float().to(device)

            outputs = model(supp_img, supp_mask, qry_img)
            pred = outputs['sim_map']               # [B,1,H,W]
            s_pred = outputs['reversed_map']
            refined_proto_list = outputs.get('refined_proto_list', None)
            triplet_loss = outputs.get('triplet_loss', torch.tensor(0.0, device=device))

            # --- threshold predictions for Dice ---
            pred_prob = pred
            pred_bin = (pred_prob > 0.5).float()

            # --- compute Dice per batch ---
            intersection = (pred_bin * qry_mask).sum(dim=(1,2,3))
            union = pred_bin.sum(dim=(1,2,3)) + qry_mask.sum(dim=(1,2,3))
            dice = (2 * intersection + 1e-6) / (union + 1e-6)
            batch_dice = dice.mean().item()
            total_dice += batch_dice
            loss_stats['dice'] += batch_dice

            # --- compute loss ---
            if use_sota_loss:
                loss, batch_stats = sota_segmentation_loss(
                    pred, s_pred, qry_mask, supp_mask,
                    refined_proto_list, None,
                    triplet_loss,
                    lambda_dice=0.5,
                    lambda_focal=1.0,
                    lambda_proto=0.1,
                    lambda_div=0.15
                )
                # Accumulate available stats
                for k, v in batch_stats.items():
                    if k in loss_stats:
                        loss_stats[k] += v
            else:
                loss = dice_loss(pred_prob, qry_mask)

            total_loss += float(loss.detach().cpu().item())

    # --- averages ---
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches

    if use_sota_loss:
        for k in loss_stats:
            loss_stats[k] /= num_batches
        print("\n===== Evaluation Breakdown =====")
        print(f"Loss components: {loss_stats}")
        print(f"Average Dice: {avg_dice:.4f}")

    return avg_dice
