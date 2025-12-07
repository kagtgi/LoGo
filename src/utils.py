import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import cv2
import math

# Import constants from dataset if available
try:
    from .dataset import CHAOS_ORGAN_LABELS, SABS_ORGAN_LABELS
except ImportError:
    # If package relative import fails (e.g. if script run directly)
    # Define fallback or expect them to be passed
    CHAOS_ORGAN_LABELS = {0: "Background", 1: "Liver", 2: "Right Kidney", 3: "Left Kidney", 4: "Spleen"}
    SABS_ORGAN_LABELS = {
        0: "Background", 1: "Spleen", 2: "Right Kidney", 3: "Left Kidney", 4: "Gallbladder",
        5: "Esophagus", 6: "Liver", 7: "Stomach", 8: "Aorta", 9: "Inferior Vena Cava",
        10: "Portal Vein and Splenic Vein", 11: "Pancreas", 12: "Right Adrenal Gland", 13: "Left Adrenal Gland"
    }

# -----------------------------
# Global plotting style
# -----------------------------
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['axes.edgecolor'] = '#D9D9D9'
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Roboto']
mpl.rcParams['axes.titleweight'] = 'medium'
mpl.rcParams['axes.titlepad'] = 12

def plot_samples(batch, n=3):
    """Plot n samples from batch."""
    support_imgs = batch["support_image"].cpu().numpy()
    support_masks = batch["support_mask"].cpu().numpy()
    query_imgs = batch["query_image"].cpu().numpy()
    query_masks = batch["query_mask"].cpu().numpy()
    organs = batch["organ_label"].cpu().numpy()

    # Combine both label mappings
    all_organ_labels = {**CHAOS_ORGAN_LABELS, **SABS_ORGAN_LABELS}

    plt.figure(figsize=(10, 4 * n))
    for i in range(min(n, len(support_imgs))):
        s_img = support_imgs[i, 0]
        s_mask = support_masks[i, 0]
        q_img = query_imgs[i, 0]
        q_mask = query_masks[i, 0]
        organ_name = all_organ_labels.get(int(organs[i]), f"Organ {int(organs[i])}")

        plt.subplot(n, 4, 4 * i + 1)
        plt.imshow(s_img, cmap="gray")
        plt.title(f"Support Image\n{organ_name}")
        plt.axis("off")

        plt.subplot(n, 4, 4 * i + 2)
        plt.imshow(s_img, cmap="gray")
        plt.imshow(s_mask, cmap="Reds", alpha=0.5)
        plt.title("Support Mask Overlay")
        plt.axis("off")

        plt.subplot(n, 4, 4 * i + 3)
        plt.imshow(q_img, cmap="gray")
        plt.title("Query Image")
        plt.axis("off")

        plt.subplot(n, 4, 4 * i + 4)
        plt.imshow(q_img, cmap="gray")
        plt.imshow(q_mask, cmap="Reds", alpha=0.5)
        plt.title("Query Mask Overlay")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def print_range(name, tensor):
    arr = tensor.cpu().numpy()
    print(f"{name}: min={arr.min():.6f}, max={arr.max():.6f}, dtype={arr.dtype}")

def visualize_samples(dataloader, n=5):
    """
    Visualize support/query images and masks from a few-shot dataloader.

    Args:
        dataloader: PyTorch DataLoader returning dict with keys:
                    "support_image", "support_mask", "query_image", "query_mask"
        n: number of batches to visualize
    """
    for i, batch in enumerate(dataloader):
        if i >= n:
            break

        support_img = batch["support_image"]   # [B,1,H,W] or [B,3,H,W]
        support_mask = batch["support_mask"]   # [B,1,H,W]
        query_img = batch["query_image"]       # [B,1,H,W] or [B,3,H,W]
        query_mask = batch["query_mask"]       # [B,1,H,W]

        B = support_img.shape[0]

        for b in range(B):
            plt.figure(figsize=(16,4))

            # Ensure images are single-channel for imshow if needed
            def to_display(img):
                if img.shape[0] == 3:  # [3,H,W] -> grayscale mean for visualization
                    return img.cpu().mean(0)
                return img.cpu()[0]    # [1,H,W] -> [H,W]

            # 1. Support image
            plt.subplot(1,4,1)
            plt.imshow(to_display(support_img[b]), cmap="gray")
            plt.title("Support Image")
            plt.axis("off")

            # 2. Support mask overlay
            plt.subplot(1,4,2)
            plt.imshow(to_display(support_img[b]), cmap="gray")
            plt.imshow(support_mask[b,0].cpu(), cmap="Reds", alpha=0.5)
            plt.title("Support Mask")
            plt.axis("off")

            # 3. Query image
            plt.subplot(1,4,3)
            plt.imshow(to_display(query_img[b]), cmap="gray")
            plt.title("Query Image")
            plt.axis("off")

            # 4. Query mask overlay
            plt.subplot(1,4,4)
            plt.imshow(to_display(query_img[b]), cmap="gray")
            plt.imshow(query_mask[b,0].cpu(), cmap="Blues", alpha=0.5)
            plt.title("Query Mask")
            plt.axis("off")

            plt.show()

def plot_distill_losses(global_losses, local_losses, save_path=None):
    # convert to cpu numpy if tensors
    global_losses = [x.detach().cpu().item() if hasattr(x, "detach") else x for x in global_losses]
    local_losses  = [x.detach().cpu().item() if hasattr(x, "detach") else x for x in local_losses]

    # total = global + local
    total_losses = [g + l for g, l in zip(global_losses, local_losses)]

    epochs = list(range(1, len(global_losses) + 1))

    plt.figure(figsize=(10, 6), dpi=300)

    plt.plot(epochs, global_losses, linewidth=2.2,
             label="Global Loss", color="#4A90E2")
    plt.plot(epochs, local_losses, linewidth=2.2,
             label="Local Loss", color="#7FCC7F")
    plt.plot(epochs, total_losses, linewidth=2.2,
             label="Total Loss", color="#E24A4A")  # New red line

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(frameon=False, fontsize=11)
    #plt.title("Distillation Loss Trend", fontsize=15, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"📁 Saved loss plot to {save_path}")

    plt.show()

def visualize_validation_predictions(model, dataloader, device, n=5):
    """
    Visualize model predictions on validation set with connected-component refinement.
    """
    model.eval()
    shown = 0

    with torch.no_grad():
        for batch in dataloader:
            if shown >= n:
                break

            support_img = batch["support_image"].to(device)
            support_mask = batch["support_mask"].float().to(device)
            query_img = batch["query_image"].to(device)
            query_mask = batch["query_mask"].float().to(device)

            outputs = model(support_img, support_mask, query_img)
            sim_map = outputs["sim_map"]  # [B, 2, H, W], Assuming this is ALREADY Softmax/Probabilities

            batch_size = sim_map.shape[0]

            # -----------------------------
            # Connected Component Refinement
            # -----------------------------
            refined_maps = []
            for b in range(batch_size):
                # 1. Get raw probabilities and hard predictions
                foreground_prob = sim_map[b, 1]
                pred_labels = torch.argmax(sim_map[b], dim=0)

                # 2. Convert to numpy for CCA
                pred_np = pred_labels.cpu().numpy().astype(np.uint8)
                # Ensure binary (just in case)
                foreground_mask = (pred_np == 1).astype(np.uint8)

                num_labels, labels_map = cv2.connectedComponents(foreground_mask, connectivity=8)

                # If nothing detected or only 1 component, skip refinement
                if num_labels <= 2: # 0 is bg, 1 is the first component. If num_labels=2, there is only 1 object.
                    refined_maps.append(sim_map[b:b+1])
                    continue

                best_component = 0
                max_score = -1.0

                # Pre-calculate total area if you want to match the exact reference formula
                # But for ranking, we only need the numerator (Mass).
                # total_area = foreground_mask.sum()

                for component_id in range(1, num_labels):
                    # Create a boolean mask for this specific component
                    component_mask = (labels_map == component_id)

                    # Get probabilities falling ONLY inside this component
                    # We don't need to convert the mask to tensor yet if we index numpy-to-tensor
                    # But for consistency with your device:
                    component_mask_tensor = torch.from_numpy(component_mask).to(sim_map.device)

                    # ---------------------------------------------------------
                    # THE FIX: Calculate Probability Mass (Sum), not Average
                    # ---------------------------------------------------------
                    # Sum of probabilities of all pixels in this component
                    score = foreground_prob[component_mask_tensor].sum()

                    # NOTE: If you strictly want the formula: Sum / (Total_Area + epsilon)
                    # score = score / (total_area + 1e-6)
                    # Mathematically, since Total_Area is constant for all components in this loop,
                    # comparing the Sum is identical to comparing the Fractions.

                    if score > max_score:
                        max_score = score
                        best_component = component_id

                # 3. Create the Refined Masks
                refined_mask = torch.from_numpy(labels_map == best_component).float().to(sim_map.device)
                inverse_mask = 1.0 - refined_mask

                # 4. Update the Sim Map
                # We want to force non-best pixels to be Background (Channel 0 = 1.0, Channel 1 = 0.0)
                refined_sim_map = sim_map[b].clone()

                # Foreground channel: Keep only the best component
                refined_sim_map[1] = refined_sim_map[1] * refined_mask

                # Background channel:
                # Logic: If pixel is in refined_mask, keep original bg prob (usually low).
                # If pixel is NOT in refined_mask (it was noise), set bg prob to 1.0.
                refined_sim_map[0] = refined_sim_map[0] * refined_mask + inverse_mask

                refined_maps.append(refined_sim_map.unsqueeze(0))

            # Re-stack
            sim_map = torch.cat(refined_maps, dim=0)

            # ---------------------------------------------------------
            # THE FIX: Do not Softmax again if input was already Softmax
            # ---------------------------------------------------------
            pred_softmax = sim_map # It remains probabilities
            pred_mask = torch.argmax(pred_softmax, dim=1)

            # -----------------------------
            # Minimalist Visualization
            # -----------------------------
            B = support_img.shape[0]
            for b in range(B):
                if shown >= n:
                    break

                fig, axes = plt.subplots(1, 5, figsize=(22, 5))
                plt.subplots_adjust(wspace=0.3)

                titles = [
                    "Support + Mask",
                    "Query Image",
                    "Ground Truth",
                    "Predicted Prob (FG)",
                    "Refined Pred Mask",
                ]

                # Panel function
                def draw_panel(ax, base, overlay=None, overlay_color="#4A90E2"):
                    ax.imshow(base, cmap="gray")
                    if overlay is not None:
                        overlay_rgba = np.zeros((*overlay.shape, 4))
                        overlay_rgba[..., 0] = int(overlay_color[1:3], 16)/255
                        overlay_rgba[..., 1] = int(overlay_color[3:5], 16)/255
                        overlay_rgba[..., 2] = int(overlay_color[5:7], 16)/255
                        overlay_rgba[..., 3] = overlay * 0.45
                        ax.imshow(overlay_rgba)
                    ax.set_xticks([]); ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#D9D9D9")
                        spine.set_linewidth(1.0)

                draw_panel(axes[0], support_img[b,0].cpu(), support_mask[b,0].cpu(), "#7FCC7F")  # soft green for support mask
                draw_panel(axes[1], query_img[b,0].cpu())
                draw_panel(axes[2], query_img[b,0].cpu(), query_mask[b,0].cpu(), "#4A90E2")      # GT in soft blue
                draw_panel(axes[3], query_img[b,0].cpu(), pred_softmax[b,1].cpu(), "#FF9E9E")    # Pred prob FG
                draw_panel(axes[4], query_img[b,0].cpu(), pred_mask[b].cpu().float(), "#FF9E9E") # Refined pred mask

                for ax, title in zip(axes, titles):
                    ax.set_title(title, fontsize=14, fontweight="medium", color="#333333")

                plt.tight_layout()
                plt.show()
                shown += 1


def visualize_feature_space_with_prototypes(model, dataloader, device, epoch, n_samples=3):
    """
    Minimalist TSNE plot of feature embeddings with prototype refinement arrows.
    Clean technical-documentation visual style.
    """
    # Soft, professional palette
    COLOR_FG = "#4A90E2"       # light blue
    COLOR_BG = "#999999"       # soft gray
    COLOR_FG_PROTO = "#003E7E" # dark blue
    COLOR_BG_PROTO = "#555555" # dark gray

    model.eval()
    sample_idx = 0

    with torch.no_grad():
        for batch in dataloader:
            if sample_idx >= n_samples:
                break

            b = 0
            support_img = batch["support_image"][b:b+1].to(device)
            support_mask = batch["support_mask"][b:b+1].float().to(device)
            query_img = batch["query_image"][b:b+1].to(device)

            outputs = model(support_img, support_mask, query_img)
            support_feats = outputs["support_feat"]

            proto_list = outputs["proto_list"]
            refined_proto_list = outputs["refined_proto_list"]

            B, C, H, W = support_feats.shape

            def flatten_feat(x):
                return x.permute(0,2,3,1).reshape(-1,C).cpu().numpy()

            support_feats_flat = flatten_feat(support_feats)

            mask = F.interpolate(support_mask, size=(H,W), mode="nearest")
            mask_flat = mask.permute(0,2,3,1).reshape(-1).cpu().numpy()

            fg_features = support_feats_flat[mask_flat > 0.95]
            bg_features = support_feats_flat[mask_flat <= 0.05]

            fg_proto, bg_proto = proto_list[0]
            fg_proto_refined, bg_proto_refined = refined_proto_list[0]

            fg_proto_all = fg_proto.detach().cpu().numpy() if fg_proto.numel() > 0 else np.zeros((0,C))
            bg_proto_all = bg_proto.detach().cpu().numpy() if bg_proto.numel() > 0 else np.zeros((0,C))
            fg_proto_refined_all = fg_proto_refined.detach().cpu().numpy() if fg_proto_refined.numel() > 0 else np.zeros((0,C))
            bg_proto_refined_all = bg_proto_refined.detach().cpu().numpy() if bg_proto_refined.numel() > 0 else np.zeros((0,C))

            # Subsample for clarity
            max_per_class = 1200
            if len(fg_features) > max_per_class:
                fg_features = fg_features[np.random.choice(len(fg_features), max_per_class, replace=False)]
            if len(bg_features) > max_per_class:
                bg_features = bg_features[np.random.choice(len(bg_features), max_per_class, replace=False)]

            if fg_proto_all.shape[0] == 0 or bg_proto_all.shape[0] == 0:
                sample_idx += 1
                continue

            combined = np.concatenate(
                [fg_features, bg_features,
                 fg_proto_all, bg_proto_all,
                 fg_proto_refined_all, bg_proto_refined_all],
                axis=0
            )

            tsne = TSNE(
                n_components=2,
                metric="cosine",
                random_state=42,
                perplexity=min(30, combined.shape[0]-1),
                n_iter=1000,
                learning_rate="auto",
                init="random"
            )
            proj = tsne.fit_transform(combined)

            n_fg, n_bg = len(fg_features), len(bg_features)
            n_fg_p, n_bg_p = len(fg_proto_all), len(bg_proto_all)
            n_fg_pr, n_bg_pr = len(fg_proto_refined_all), len(bg_proto_refined_all)

            fg_proj = proj[:n_fg]
            bg_proj = proj[n_fg:n_fg+n_bg]

            offset = n_fg + n_bg
            fg_p_proj = proj[offset:offset+n_fg_p]
            bg_p_proj = proj[offset+n_fg_p:offset+n_fg_p+n_bg_p]
            fg_pr_proj = proj[offset+n_fg_p+n_bg_p:offset+n_fg_p+n_bg_p+n_fg_pr]
            bg_pr_proj = proj[offset+n_fg_p+n_bg_p+n_fg_pr:]

            # -----------------------------
            # Clean Minimalist Scatter Plot
            # -----------------------------
            plt.figure(figsize=(9,7), dpi=300)

            # Main features (very soft colors)
            plt.scatter(bg_proj[:,0], bg_proj[:,1], s=6, c=COLOR_BG, alpha=0.25, label="BG Features")
            plt.scatter(fg_proj[:,0], fg_proj[:,1], s=6, c=COLOR_FG, alpha=0.25, label="FG Features")

            # Arrows for refinement (thin, geometric)
            for i in range(min(len(fg_p_proj), len(fg_pr_proj))):
                plt.arrow(
                    fg_p_proj[i,0], fg_p_proj[i,1],
                    fg_pr_proj[i,0] - fg_p_proj[i,0],
                    fg_pr_proj[i,1] - fg_p_proj[i,1],
                    width=0.08,
                    head_width=0.9,
                    head_length=0.9,
                    color=COLOR_FG_PROTO,
                    alpha=0.55,
                    length_includes_head=True
                )

            for i in range(min(len(bg_p_proj), len(bg_pr_proj))):
                plt.arrow(
                    bg_p_proj[i,0], bg_p_proj[i,1],
                    bg_pr_proj[i,0] - bg_p_proj[i,0],
                    bg_pr_proj[i,1] - bg_p_proj[i,1],
                    width=0.08,
                    head_width=0.9,
                    head_length=0.9,
                    color=COLOR_BG_PROTO,
                    alpha=0.55,
                    length_includes_head=True
                )

            # Prototype markers (clean, bold)
            plt.scatter(fg_p_proj[:,0], fg_p_proj[:,1], s=110,
                        c=COLOR_FG, marker="o", edgecolors="#333333",
                        linewidths=1.0, label="FG Proto (Init)", zorder=4)

            plt.scatter(bg_p_proj[:,0], bg_p_proj[:,1], s=110,
                        c=COLOR_BG, marker="o", edgecolors="#333333",
                        linewidths=1.0, label="BG Proto (Init)", zorder=4)

            plt.scatter(fg_pr_proj[:,0], fg_pr_proj[:,1], s=130,
                        c=COLOR_FG_PROTO, marker="X", edgecolors="#000000",
                        linewidths=1.0, label="FG Proto (Refined)", zorder=5)

            plt.scatter(bg_pr_proj[:,0], bg_pr_proj[:,1], s=130,
                        c=COLOR_BG_PROTO, marker="X", edgecolors="#000000",
                        linewidths=1.0, label="BG Proto (Refined)", zorder=5)

            # Titles & labels
            plt.title(
                f"T-SNE Feature Space — Epoch {epoch}, Sample {sample_idx+1}",
                fontsize=15, fontweight="medium", pad=10, color="#222222"
            )
            plt.xlabel("TSNE Dim 1", fontsize=12, color="#333333")
            plt.ylabel("TSNE Dim 2", fontsize=12, color="#333333")

            # Legend (clean box)
            plt.legend(
                frameon=True,
                fontsize=10,
                facecolor="white",
                edgecolor="#D9D9D9"
            )

            plt.grid(True, linestyle="--", alpha=0.25)
            plt.tight_layout()
            plt.show()

            sample_idx += 1


def print_prototype_stats(proto_list, refined_proto_list, epoch):
    """Print statistics about prototypes for debugging."""
    print(f"\n[Epoch {epoch}] Prototype Statistics:")
    print("=" * 60)
    # Placeholder for actual stats printing logic if needed
