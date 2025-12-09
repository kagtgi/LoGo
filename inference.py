import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from src.dataset import create_unified_dataloaders
from src.model import LoGoEncoder, ProtoSegNet

def parse_args():
    parser = argparse.ArgumentParser(description="Inference / Demo for LoGo")
    
    parser.add_argument("--chaos_dir", type=str, default="./chaos_MR_T2_normalized/chaos_MR_T2_normalized")
    parser.add_argument("--sabs_dir", type=str, default="./sabs_CT_normalized/sabs_CT_normalized")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ProtoSegNet checkpoint")
    parser.add_argument("--output_dir", type=str, default="./inference_results")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--device", type=str, default="cuda")
    
    return parser.parse_args()

def save_prediction_plot(sample_idx, support_img, support_mask, query_img, query_mask, pred_prob, pred_mask, output_dir):
    """
    Saves a visualization panel for a single sample.
    """
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    plt.subplots_adjust(wspace=0.3)
    
    titles = [
        "Support + Mask",
        "Query Image",
        "Ground Truth",
        "Predicted Prob (FG)",
        "Refined Pred Mask",
    ]

    def to_cpu(x):
        return x.detach().cpu().numpy()

    # Helpers for display
    s_img = to_cpu(support_img[0][0])
    s_mask = to_cpu(support_mask[0][0])
    q_img = to_cpu(query_img[0][0])
    q_mask = to_cpu(query_mask[0][0])
    p_prob = to_cpu(pred_prob)
    p_mask = to_cpu(pred_mask)

    def draw_panel(ax, base, overlay=None, overlay_color="#4A90E2"):
        ax.imshow(base, cmap="gray")
        if overlay is not None:
            overlay_rgba = np.zeros((*overlay.shape, 4))
            # Hex to RGB
            color = tuple(int(overlay_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            overlay_rgba[..., 0] = color[0]/255
            overlay_rgba[..., 1] = color[1]/255
            overlay_rgba[..., 2] = color[2]/255
            overlay_rgba[..., 3] = overlay * 0.45
            ax.imshow(overlay_rgba)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#D9D9D9")
            spine.set_linewidth(1.0)

    draw_panel(axes[0], s_img, s_mask, "#7FCC7F")
    draw_panel(axes[1], q_img)
    draw_panel(axes[2], q_img, q_mask, "#4A90E2")
    draw_panel(axes[3], q_img, p_prob, "#FF9E9E")
    draw_panel(axes[4], q_img, p_mask, "#FF9E9E")

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=14, fontweight="medium", color="#333333")

    save_path = os.path.join(output_dir, f"sample_{sample_idx}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved inference result to {save_path}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    # We use validation loader for inference demo
    print("Loading data...")
    _, _, val_loader = create_unified_dataloaders(
        chaos_dir=args.chaos_dir,
        sabs_dir=args.sabs_dir,
        batch_size=1,
        num_workers=0
    )
    
    # 2. Load Model
    print(f"Loading model from {args.checkpoint}...")
    encoder = LoGoEncoder(device=str(device))
    model = ProtoSegNet(encoder, use_refiner=True).to(device)
    
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return

    model.eval()

    print(f"Running inference on {args.n_samples} samples...")
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.n_samples:
                break
                
            supp_img = batch["support_image"].to(device)
            supp_mask = batch["support_mask"].float().to(device)
            qry_img = batch["query_image"].to(device)
            qry_mask = batch["query_mask"].float().to(device) # For Visualization only

            # Forward
            outputs = model(supp_img, supp_mask, qry_img)
            sim_map = outputs["sim_map"] # [B, 2, H, W] logits OR probabilities?
            
            # The model output is declared as LOGITS in model.py comments but applied Softmax at the end?
            # Let's check model.py:
            # "sim_map_logits = F.softmax(sim_map_logits, dim=1)" -> It is actually probabilities!
            # Then interpolated. So sim_map is probabilities [0..1]
            
            pred_prob = sim_map[0, 1] # FG Probability
            pred_mask = torch.argmax(sim_map[0], dim=0) # [H, W]

            # Post-processing (CCA) - optional, copied from visualization util logic
            # Skipping CCA for this simple inference script to keep it clean, 
            # effectively showing raw network output.

            save_prediction_plot(
                i, supp_img, supp_mask, qry_img, qry_mask,
                pred_prob, pred_mask, args.output_dir
            )

    print("Done.")

if __name__ == "__main__":
    main()
