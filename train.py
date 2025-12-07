import argparse
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from src.dataset import create_unified_dataloaders
from src.model import LoGoEncoder, ProtoSegNet
from src.train_utils import train_one_epoch, evaluate
from src.utils import plot_distill_losses # We might want a generic plot loss
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoGo ProtoSegNet")
    
    # Data params
    parser.add_argument("--chaos_dir", type=str, default="./chaos_MR_T2_normalized/chaos_MR_T2_normalized", help="Path to CHAOS dataset")
    parser.add_argument("--sabs_dir", type=str, default="./sabs_CT_normalized/sabs_CT_normalized", help="Path to SABS dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    
    # Model params
    parser.add_argument("--distilled_weights", type=str, default="./distilled_logoencoder.pt", help="Path to distilled encoder weights")
    parser.add_argument("--use_refiner", action="store_true", default=True, help="Use prototype refiner")
    
    # Training params
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--workers", type=int, default=0, help="Number of data loader workers")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loaders
    print("Creating dataloaders...")
    try:
        train_loader, _, val_loader = create_unified_dataloaders(
            chaos_dir=args.chaos_dir,
            sabs_dir=args.sabs_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            use_both_train=True,
            use_both_val=True
        )
    except Exception as e:
        print(f"❌ Error creating dataloaders: {e}")
        return

    # Initialize Model
    print("Initializing LoGo Encoder...")
    encoder = LoGoEncoder(device=str(device))
    
    # Load distilled weights if available
    if os.path.exists(args.distilled_weights):
        print(f"✅ Loading distilled weights from {args.distilled_weights}")
        try:
            encoder.load_weights(args.distilled_weights, map_location=device)
        except Exception as e:
            print(f"⚠️ Failed to load weights: {e}")
    else:
        print("⚠️ Distilled weights not found. Training from scratch initialization.")

    # Wrap in ProtoSegNet
    model = ProtoSegNet(encoder, use_refiner=args.use_refiner).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) # Added weight decay
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.5)

    print(f"Starting training for {args.epochs} epochs...")
    
    best_dice = 0.0
    train_losses = []
    val_dices = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)
        train_losses.append(avg_loss)
        print(f"  Train Loss: {avg_loss:.4f}")
        
        # Validation
        avg_dice = evaluate(model, val_loader, device)
        val_dices.append(avg_dice)
        print(f"  Val Dice: {avg_dice:.4f}")
        
        scheduler.step()
        
        # Save Best
        if avg_dice > best_dice:
            best_dice = avg_dice
            save_path = os.path.join(args.output_dir, "protosegnet_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  ⭐ New best Dice! Saved to {save_path}")
            
        # Save Last
        torch.save(model.state_dict(), os.path.join(args.output_dir, "protosegnet_last.pt"))

    print("\nTraining complete.")
    print(f"Best Validation Dice: {best_dice:.4f}")
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_dices, label='Val Dice', color='orange')
    plt.title('Validation Dice')
    plt.legend()
    
    plot_path = os.path.join(args.output_dir, "training_curves.png")
    plt.savefig(plot_path)
    print(f"Saved training curves to {plot_path}")

if __name__ == "__main__":
    main()
