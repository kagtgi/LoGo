import argparse
import os
import torch
import torch.optim as optim
from src.dataset import create_unified_dataloaders
from src.model import LoGoEncoder
from src.train_utils import distill_one_epoch
from src.utils import plot_distill_losses

def parse_args():
    parser = argparse.ArgumentParser(description="Distill LoGo Encoder")
    
    # Data params
    parser.add_argument("--chaos_dir", type=str, default="./chaos_MR_T2_normalized/chaos_MR_T2_normalized", help="Path to CHAOS dataset")
    parser.add_argument("--sabs_dir", type=str, default="./sabs_CT_normalized/sabs_CT_normalized", help="Path to SABS dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    
    # Training params
    parser.add_argument("--epochs", type=int, default=60, help="Number of distillation epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--workers", type=int, default=0, help="Number of data loader workers")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create Dataloaders
    try:
        _, distill_loader, _ = create_unified_dataloaders(
            chaos_dir=args.chaos_dir,
            sabs_dir=args.sabs_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            use_both_train=True,
            use_both_val=True
        )
    except Exception as e:
        print(f"❌ Failed to create dataloaders: {e}")
        return

    # Initialize Model
    encoder = LoGoEncoder(device=str(device)).to(device)
    
    # Optimizer & Scheduler
    distill_optimizer = optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=0.0005)
    
    # Estimate steps
    n_steps = len(distill_loader) * args.epochs
    lr_milestones = [(ii + 1) * 1000 for ii in range(n_steps // 1000 - 1)]
    distill_scheduler = optim.lr_scheduler.MultiStepLR(
        distill_optimizer,
        milestones=lr_milestones,
        gamma=0.95
    )

    global_losses = []
    local_losses = []

    print(f"Starting distillation for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        loss_global, loss_local = distill_one_epoch(encoder, distill_loader, distill_optimizer, device)
        
        global_losses.append(loss_global)
        local_losses.append(loss_local)
        
        print(f"  Global Loss: {loss_global:.4f}, Local Loss: {loss_local:.4f}")
        
        distill_scheduler.step()
        
        # Save checkpoint periodically
        if epoch % 10 == 0:
             encoder.save_weights(os.path.join(args.output_dir, f"logoencoder_epoch_{epoch}.pt"))

    # Save weights
    final_path = os.path.join(args.output_dir, "distilled_logoencoder.pt")
    encoder.save_weights(final_path)
    print(f"✅ Final weights saved to {final_path}")
    
    # Plot results
    plot_path = os.path.join(args.output_dir, "distill_loss_curve.png")
    plot_distill_losses(global_losses, local_losses, plot_path)

if __name__ == "__main__":
    main()
