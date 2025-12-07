import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.segmentation import deeplabv3_resnet101
from axial_attention import AxialAttention, AxialPositionalEmbedding
import math
import os
import random

# ==============================
# Helper Functions
# ==============================
def safe_norm(x, p=2, dim=1, eps=1e-8):
    """Safely normalize x along dim to unit length."""
    return x / (x.norm(p=p, dim=dim, keepdim=True).clamp(min=eps))

# ==============================
# Pretrained Encoder
# ==============================
class PretrainedEncoder(nn.Module):
    def __init__(self, backbone="resnet18", feature_dim=256, pretrained=True):
        super().__init__()
        self.feature_dim = feature_dim

        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            layers = list(base.children())[:-2]
            self.encoder = nn.Sequential(*layers)
            in_channels = 512

        elif backbone == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            layers = list(base.children())[:-2]
            self.encoder = nn.Sequential(*layers)
            in_channels = 2048

        elif backbone == "vgg16":
            base = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None)
            self.encoder = base.features
            in_channels = 512

        else:
            raise ValueError(f"Backbone {backbone} not supported.")

        # Patch first conv layer to 1 channel
        old_conv = self.encoder[0]
        self.encoder[0] = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        with torch.no_grad():
            self.encoder[0].weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        self.proj = nn.Conv2d(in_channels, feature_dim, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.proj(feat)
        return feat

# ==============================
# LoGo Encoder
# ==============================
class ConvFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.fuse_fc = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_loc, use_conv=True):
        """
        x: [B, C] or [B, C, H, W]
        x_loc: [B, C] or [B, C, H, W]
        """
        if x.dim() == 4 and x_loc.dim() == 4:
            # Both are [B, C, H, W]
            return self.fuse_conv(torch.cat([x, x_loc], dim=1))
        else:
            # Both are [B, C] - use FC layer
            return self.fuse_fc(torch.cat([x, x_loc], dim=1))


# RMSNorm implementation (faster alternative to LayerNorm)
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - used in modern transformers"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        # x: [B, C, H, W]
        norm = torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return x * norm * self.weight


class LoGoEncoder(nn.Module):
    def __init__(self, device='cuda'):
        super(LoGoEncoder, self).__init__()

        self.device = device

        # Global pathway - processes full image
        # Using GroupNorm (more stable than BatchNorm) + GELU activation
        self.conv1 = nn.Sequential(
            # Layer 1: 256×256 → 128×128
            nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 128),  # 32 groups for 128 channels
            nn.GELU(),
            nn.Dropout(0.5),

            # Layer 2: 128×128 → 64×64
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.Dropout(0.5),

            # Layer 3: 64×64 → 32×32
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.Dropout(0.5),
        )

        # Local pathway - processes patches
        self.conv1_p = nn.Sequential(
            # Layer 1: 64×64 → 32×32
            nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.GELU(),
            nn.Dropout(0.5),

            # Layer 2: 32×32 → 16×16
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.Dropout(0.5),

            # Layer 3: 16×16 → 8×8
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.Dropout(0.5),
        )

        # Global attention block with RMSNorm (faster than LayerNorm)
        self.global_block = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            RMSNorm(512),
            AxialPositionalEmbedding(dim=512, shape=(32, 32)),
            AxialAttention(dim=512, heads=16, dim_index=1),
            AxialPositionalEmbedding(dim=512, shape=(32, 32)),
            AxialAttention(dim=512, heads=8, dim_index=1),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            RMSNorm(512),
        )

        # Cross-patch attention block with RMSNorm
        self.cross_patch_block = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            RMSNorm(512),
            AxialPositionalEmbedding(dim=512, shape=(32, 32)),
            AxialAttention(dim=512, heads=8, dim_index=1),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            RMSNorm(512),
        )

        # Local attention block with RMSNorm
        self.local_block = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            RMSNorm(512),
        )

        self.local_pos_embed = AxialPositionalEmbedding(dim=512, shape=(8, 8))
        self.local_attention = AxialAttention(dim=512, heads=16, dim_index=1)
        self.local_pos_embed2 = AxialPositionalEmbedding(dim=512, shape=(8, 8))
        self.local_attention2 = AxialAttention(dim=512, heads=8, dim_index=1)
        self.local_block_post = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            RMSNorm(512),
        )

        # Gated fusion with Swish activation
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.SiLU()  # Swish/SiLU for smoother gating
        )

        # Fusion and projection layers with GroupNorm
        self.adjust_p = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(32, 1024),
            nn.GELU(),
            nn.Dropout(0.25)
        )

        self.weights = nn.Parameter(torch.tensor([0.5, 0.5]), requires_grad=True)

        # Load pretrained weights
        self.global_teacher = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        self.global_teacher.to(self.device)

        # Local Teacher: DeepLabv3 with ResNet101 backbone
        self.local_teacher = deeplabv3_resnet101(pretrained=True)
        self.local_teacher.to(self.device)

        self.Tg = 4

        # Projection layers with proper initialization
        self.proj_teacher_local = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj_teacher_local.weight, mode="fan_out", nonlinearity="relu")

        weight_path = "./logoencoder_final.pt"
        if os.path.exists(weight_path):
            try:
                self.load_state_dict(torch.load(weight_path, map_location=self.device))
                print(f"✅ Loaded existing weights from {weight_path}")
            except Exception as e:
                print(f"⚠️ Failed to load weights from {weight_path}: {e}")
        else:
            print("ℹ️ No existing weights found, starting from scratch.")

    def forward(self, x):
        # Handle single-channel inputs (convert to 3-channel)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"Unexpected channel size: {x.shape}")

        img_size = x.shape[-1]
        xin = x
        B = x.shape[0]

        # ==================== Global Pathway ====================
        x = self.conv1(x)  # [B, 512, 32, 32] for 256×256 input
        x_attn = self.global_block(x)
        x = x + x_attn  # [B, 512, 32, 32]

        H, W = x.shape[2], x.shape[3]  # Should be 32×32

        # ==================== Local Pathway with Reassembly ====================
        patch_size = 64  # Original image patch size
        grid_h = img_size // patch_size  # 256 // 64 = 4
        grid_w = img_size // patch_size  # 256 // 64 = 4

        feat_patch_size = 8

        # Initialize output tensor for reassembled patches
        x_loc = torch.zeros(B, 512, grid_h * feat_patch_size, grid_w * feat_patch_size,
                           device=x.device, dtype=x.dtype)

        # Process each patch in grid
        for i in range(grid_h):
            for j in range(grid_w):
                x_p_init = xin[:, :,
                         patch_size * i : patch_size * (i+1),
                         patch_size * j : patch_size * (j+1)]  # [B, 3, 64, 64]

                # Process patch through local pathway
                x_p_init = self.conv1_p(x_p_init)  # [B, 512, 8, 8]

                # Apply local attention blocks
                x_p = self.local_block(x_p_init)
                x_p = self.local_pos_embed(x_p)
                x_p = self.local_attention(x_p)
                x_p = self.local_pos_embed2(x_p)
                x_p = self.local_attention2(x_p)
                x_p = self.local_block_post(x_p)  # [B, 512, 8, 8]

                # Place processed patch back into correct position
                x_loc[:, :,
                     feat_patch_size * i : feat_patch_size * (i+1),
                     feat_patch_size * j : feat_patch_size * (j+1)] = x_p

        # ==================== Fusion with Adaptive Instance Normalization ====================
        # Using AdaIN for more flexible feature alignment
        x_loc = x + self.cross_patch_block(x_loc)

        x = self.adaptive_instance_norm(x)
        x_loc = self.adaptive_instance_norm(x_loc)

        gate = self.fuse_gate(torch.cat([x, x_loc], dim=1))   # [B, 512, 32, 32]

        # Weighted combination of global and local features
        x_combine = gate * x + (1 - gate) * x_loc

        # ==================== Residual connection with global features ====================
        x_combine = torch.cat([x_combine, x], dim=1)  # [B, 1024, 32, 32]

        # ==================== Final projection ====================
        x_combine = self.adjust_p(x_combine)  # [B, 1024, 32, 32]

        return x, x_loc, x_combine

    def adaptive_instance_norm(self, x):
        """Adaptive Instance Normalization for better feature alignment"""
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + 1e-6
        return (x - mean) / std

    def save_weights(self, path="./logoencoder_final.pt"):
        torch.save(self.state_dict(), path)
        print(f"✅ Weights saved to {path}")

    def load_weights(self, path="./logoencoder_final.pt", map_location='cuda'):
        self.load_state_dict(torch.load(path, map_location=map_location))
        print(f"✅ Weights loaded from {path}")

    def global_distillation_loss(self, student, teacher):
        """
        student: [B, C, H, W] - student global features [5, , 32, 32]
        teacher: [B, N, D] - teacher patch tokens from DINOv2 [5, 256, 1024]
        """
        def student_to_patches(student, N=256):
            # [B, C, H, W] -> [B, N, C]
            student_feat = student.permute(0, 2, 3, 1)  # [B, H, W, C]
            B, H, W, D = student_feat.shape
            P = 8
            h_patch, w_patch = H // P, W // P
            student = student_feat.view(B, P, h_patch, P, w_patch, D)
            student = student.mean(dim=(2, 4))
            student = student.permute(0, 3, 1, 2)
            student = F.interpolate(student, size=(16, 16), mode='bilinear', align_corners=False)
            student = student.permute(0, 2, 3, 1).reshape(B, 256, D)
            return student

        def multi_pos_nce(student_patches, teacher_patches, tau=0.07):
            """
            student_patches: [B, N, D_s]
            teacher_patches: [B, N, D_t]
            """
            B, N, D_s = student_patches.shape
            _, _, D_t = teacher_patches.shape

            # Project teacher to student dimension if needed
            if D_t != D_s:
                # Simple projection: take first D_s dimensions or pad
                if D_t > D_s:
                    teacher_patches = teacher_patches[:, :, :D_s]
                else:
                    # Pad if teacher is smaller
                    pad_size = D_s - D_t
                    teacher_patches = torch.cat([
                        teacher_patches,
                        torch.zeros(B, N, pad_size, device=teacher_patches.device, dtype=teacher_patches.dtype)
                    ], dim=-1)

            s = F.normalize(student_patches, dim=-1).reshape(B * N, D_s)
            t = F.normalize(teacher_patches, dim=-1).reshape(B * N, D_s)
            logits = s @ t.T / tau
            labels = torch.arange(B * N, device=s.device)
            return F.cross_entropy(logits, labels) / math.log(B * N)

        def rkd_loss(g_s, g_t):
            B, D = g_s.shape
            if B < 2:
                return torch.tensor(0.0, device=g_s.device)
            dist_s = torch.cdist(g_s, g_s)
            dist_t = torch.cdist(g_t, g_t)
            mean_s, mean_t = dist_s[dist_s > 0].mean(), dist_t[dist_t > 0].mean()
            dist_s, dist_t = dist_s / mean_s, dist_t / mean_t
            return F.l1_loss(dist_s, dist_t)

        def cov_loss(g_s, g_t, normalize=True, use_mse=True, scale=1.0):
            if normalize:
                g_s = F.normalize(g_s, dim=-1)
                g_t = F.normalize(g_t, dim=-1)
            g_s = g_s - g_s.mean(dim=0, keepdim=True)
            g_t = g_t - g_t.mean(dim=0, keepdim=True)

            # Ensure same dimension
            min_dim = min(g_s.shape[-1], g_t.shape[-1])
            g_s = g_s[:, :min_dim]
            g_t = g_t[:, :min_dim]

            C_s = g_s.T @ g_s / (g_s.shape[0] - 1)
            C_t = g_t.T @ g_t / (g_t.shape[0] - 1)
            loss = torch.mean((C_s - C_t) ** 2) if use_mse else torch.norm(C_s - C_t, p='fro')
            return loss * scale

        def feature_decorrelation_loss(z_s, z_t, eps=1e-5):
            # Ensure same dimension
            min_dim = min(z_s.shape[-1], z_t.shape[-1])
            z_s = z_s[:, :min_dim]
            z_t = z_t[:, :min_dim]

            z_s = (z_s - z_s.mean(0)) / (z_s.std(0) + eps)
            z_t = (z_t - z_t.mean(0)) / (z_t.std(0) + eps)
            c_s = z_s.T @ z_s / z_s.shape[0]
            c_t = z_t.T @ z_t / z_t.shape[0]
            I = torch.eye(c_s.size(0), device=z_s.device)
            return (F.mse_loss(c_s, I) + F.mse_loss(c_t, I)) / 2

        def kl_divergence_loss(g_s, g_t):
            # Ensure same dimension
            min_dim = min(g_s.shape[-1], g_t.shape[-1])
            g_s = g_s[:, :min_dim]
            g_t = g_t[:, :min_dim]

            p_s = F.log_softmax(g_s, dim=-1)
            p_t = F.softmax(g_t, dim=-1)
            return F.kl_div(p_s, p_t, reduction='batchmean')

        # Adaptive weighting based on model's gradient norm
        total_grad_norm = sum(p.grad.norm().item() if p.grad is not None else 0.0
                            for p in self.parameters())
        alpha = 1.0 / (1.0 + math.exp(-0.01 * total_grad_norm))
        beta = 1.0 - alpha

        s_patches = student_to_patches(student)  # [B, 256, 512]
        g_s = s_patches.mean(dim=1)  # [B, 512]
        g_t = teacher.mean(dim=1)    # [B, 1024]

        loss_nce = multi_pos_nce(s_patches, teacher)
        loss_rkd = rkd_loss(g_s, g_t)
        loss_cov = cov_loss(g_s, g_t)
        loss_decor = feature_decorrelation_loss(g_s, g_t)
        loss_kl = kl_divergence_loss(g_s, g_t)

        total = alpha * (loss_nce + loss_rkd) + beta * (loss_cov + loss_decor + loss_kl)

        return total, {
            "NCE": loss_nce.item(),
            "RKD": loss_rkd.item(),
            "Cov": loss_cov.item(),
            "Decor": loss_decor.item(),
            "KL": loss_kl.item(),
            "Alpha": alpha,
            "Beta": beta
        }

    def local_distillation_loss(self, student, teacher,
                            w_cos=1.0, w_nce=0.5, w_edge=0.5,
                            w_perc=0.0, w_mi=0.0, w_chamfer=0.2,
                            use_pixel_nce=True, use_chamfer=True):
        def per_pixel_cosine_loss(student, teacher):
            s = F.normalize(student, dim=-1)
            t = F.normalize(teacher, dim=-1)
            return (1 - (s * t).sum(-1)).mean()

        def pixel_nce_loss(student, teacher, tau=0.07):
            B, H, W, D = student.shape
            s = F.normalize(student, dim=-1).reshape(B * H * W, D)
            t = F.normalize(teacher, dim=-1).reshape(B * H * W, D)
            logits = (s @ t.T) / tau
            labels = torch.arange(B * H * W, device=student.device)
            return F.cross_entropy(logits, labels) / math.log(max(2, B * H * W))

        def edge_aware_loss(student, teacher):
            def sobel(x):
                gx = x[:, 1:, :, :] - x[:, :-1, :, :]
                gy = x[:, :, 1:, :] - x[:, :, :-1, :]
                gx = F.pad(gx, (0, 0, 0, 0, 1, 0))
                gy = F.pad(gy, (0, 0, 1, 0, 0, 0))
                return gx, gy
            gx_s, gy_s = sobel(student)
            gx_t, gy_t = sobel(teacher)
            return F.l1_loss(gx_s, gx_t) + F.l1_loss(gy_s, gy_t)

        def local_mi_loss(student, teacher, eps=1e-8):
            B, H, W, D = student.shape
            s = F.normalize(student, dim=-1).reshape(B * H * W, D)
            t = F.normalize(teacher, dim=-1).reshape(B * H * W, D)
            logits = (s @ t.T) / 0.1
            labels = torch.arange(B * H * W, device=student.device)
            return F.cross_entropy(logits, labels) / math.log(max(2, B * H * W))

        def chamfer_loss(student, teacher, squared=True, normalize=True):
            B, H, W, D = student.shape
            s = student.reshape(B, H * W, D)
            t = teacher.reshape(B, H * W, D)
            if normalize:
                s = F.normalize(s, dim=-1)
                t = F.normalize(t, dim=-1)
            loss_total = 0.0
            for b in range(B):
                dists = torch.cdist(s[b], t[b])
                if squared:
                    dists = dists ** 2
                loss_st = dists.min(dim=1)[0].mean()
                loss_ts = dists.min(dim=0)[0].mean()
                loss_total += (loss_st + loss_ts)
            return loss_total / B

        def feature_decorrelation_loss(z_s, z_t, eps=1e-5):
            z_s = (z_s - z_s.mean(0)) / (z_s.std(0) + eps)
            z_t = (z_t - z_t.mean(0)) / (z_t.std(0) + eps)
            c_s = z_s.T @ z_s / z_s.shape[0]
            c_t = z_t.T @ z_t / z_t.shape[0]
            I = torch.eye(c_s.size(0), device=z_s.device)
            return (F.mse_loss(c_s, I) + F.mse_loss(c_t, I)) / 2

        def kl_divergence_loss(g_s, g_t):
            p_s = F.log_softmax(g_s, dim=-1)
            p_t = F.softmax(g_t, dim=-1)
            return F.kl_div(p_s, p_t, reduction='batchmean')

        with torch.no_grad():
            total_grad_norm = sum(p.grad.norm().item() for p in self.parameters() if p.grad is not None)
        alpha = 1.0 / (1.0 + math.exp(-0.01 * total_grad_norm))
        beta = 1.0 - alpha

        s = student.permute(0, 2, 3, 1)
        t = teacher.permute(0, 2, 3, 1)

        loss_cos = per_pixel_cosine_loss(s, t)
        loss_edge = edge_aware_loss(s, t)

        loss_nce = pixel_nce_loss(s, t) if use_pixel_nce else torch.tensor(0.0, device=s.device)
        loss_mi = local_mi_loss(s, t) if w_mi > 0 else torch.tensor(0.0, device=s.device)
        loss_cham = chamfer_loss(s, t) if use_chamfer else torch.tensor(0.0, device=s.device)
        loss_perc = torch.tensor(0.0, device=s.device)

        # local pooled globals
        gp_s = F.adaptive_avg_pool2d(student.permute(0, 3, 1, 2), 1).view(student.shape[0], -1)
        gp_t = F.adaptive_avg_pool2d(teacher.permute(0, 3, 1, 2), 1).view(teacher.shape[0], -1)

        loss_decor = feature_decorrelation_loss(gp_s, gp_t)
        loss_kl = kl_divergence_loss(gp_s, gp_t)

        lightweight = (w_cos * loss_cos + w_edge * loss_edge)
        heavier = (w_nce * loss_nce + w_mi * loss_mi + w_chamfer * loss_cham + w_perc * loss_perc)
        stats = {
            "cosine": loss_cos.item(),
            "edge": loss_edge.item(),
            "nce": (loss_nce.item() if isinstance(loss_nce, torch.Tensor) else float(loss_nce)),
            "mi": (loss_mi.item() if isinstance(loss_mi, torch.Tensor) else float(loss_mi)),
            "chamfer": (loss_cham.item() if isinstance(loss_cham, torch.Tensor) else float(loss_cham)),
            "decor": loss_decor.item(),
            "kl": loss_kl.item(),
            "alpha": alpha,
            "beta": beta
        }

        total = alpha * lightweight + beta * (heavier + 0.5 * loss_decor + 0.5 * loss_kl)

        return total, stats


    def distill(self, x, optimizer):
        print(f"x shape: {x.shape}")
        self.train()

        # Forward pass student
        print()
        print("STUDENT: ")
        print("=======================")
        student_global, student_local, student_combined = self.forward(x)

        print("Student global shape: ", student_global.shape)
        print("Student local shape: ", student_local.shape)
        print("Student combined shape: ", student_combined.shape)

        if x.shape[1] == 1:  # grayscale
            x_rgb = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 3:  # already RGB
            x_rgb = x
        else:
            raise ValueError(f"Unexpected channel size: {x.shape}")

        # Teacher: Global (DINOv2)
        print()
        print("GLOBAL TEACHER: ")
        print("=======================")
        t_in = F.interpolate(x_rgb, size=(224, 224), mode="bilinear", align_corners=False)
        print(f"Teacher input: {t_in.shape}")

        t_features = self.global_teacher.forward_features(t_in)
        teacher_global = t_features["x_norm_patchtokens"]

        print(f"Teacher output: {teacher_global.shape}")

        # Local Teacher (DeepLabv3)
        print()
        print("LOCAL TEACHER: ")
        print("=======================")
        features = {}
        def hook_layer3(module, input, output):
            features['layer3'] = output

        handle = self.local_teacher.backbone.layer3.register_forward_hook(hook_layer3)
        _ = self.local_teacher(x_rgb)

        teacher_local = features['layer3']  # [B, 1024, 32, 32]
        handle.remove()

        print(f"Teacher output (after projection): {teacher_local.shape}")

        # Distillation losses
        print()
        print("LOSS COMPUTATION: ")
        print("=======================")
        loss_global, loss_global_components = self.global_distillation_loss(student_combined, teacher_global)

        print(f"Global loss: {loss_global}")
        print(f"Global loss components: {loss_global_components}")

        loss_local, loss_local_components = self.local_distillation_loss(student_combined, teacher_local)

        print(f"Local loss: {loss_local}")
        print(f"Local loss components: {loss_local_components}")

        #weights = F.softmax(self.weights, dim=0)

        #loss = weights[0] * loss_global + weights[1] * loss_local
        loss = loss_global + loss_local
        print(f"Total loss: {loss}")

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss_global, loss_local

# ============================================================
# SOTA: Spherical K-Means with K-means++ Initialization
# ============================================================
def kmeans_plus_plus_init(x_norm, n_clusters):
    """
    SOTA center initialization for spherical k-means.
    x_norm: [N, C] unit-normalized vectors
    """
    device = x_norm.device
    N = x_norm.shape[0]

    # 1. Randomly pick the first center
    first_idx = torch.randint(0, N, (1,), device=device)
    centers = x_norm[first_idx].clone()

    # 2. Select remaining centers with K-means++ probability
    for _ in range(1, n_clusters):
        # Squared L2 distance to closest center
        dist_sq = torch.cdist(x_norm, centers, p=2).pow(2)
        min_dist_sq = dist_sq.min(dim=1)[0]

        # Probability proportional to distance²
        probs = min_dist_sq / (min_dist_sq.sum() + 1e-12)

        # Sample next center
        next_idx = torch.multinomial(probs, 1)
        centers = torch.cat([centers, x_norm[next_idx]], dim=0)

    return centers


# ============================================================
# Core: Spherical K-Means
# ============================================================
def kmeans_torch(x, n_clusters=10, n_iters=25, device=None):
    """
    Performs spherical k-means using cosine similarity.
    x: [num_pixels, C] features
    Returns: [K, C] unit-norm cluster centers
    """
    if x.numel() == 0:
        return torch.zeros(0, x.shape[1], device=x.device)

    device = device or x.device
    x = x.to(device)

    # Normalize input (spherical k-means uses directions only)
    x_norm = safe_norm(x, dim=1)

    n_clusters = min(n_clusters, x_norm.shape[0])

    # ------------------------------
    # ✅ SOTA K-means++ initialization
    # ------------------------------
    centers = kmeans_plus_plus_init(x_norm, n_clusters)

    for _ in range(n_iters):
        # Cosine similarity -> cluster assignment
        sim = torch.matmul(x_norm, centers.T)  # [N, K]
        labels = sim.argmax(dim=1)

        # Update cluster centers
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                cluster_feat = x_norm[mask]
                new_center = cluster_feat.mean(dim=0)
                centers[k] = safe_norm(new_center.unsqueeze(0), dim=1)[0]

    return centers

# -------------------------
# Prototype Refiner (Optimized)
# -------------------------
class PrototypeRefiner(nn.Module):
    def __init__(self, beta: float = 1.0, m: float = 0.3, lr_refine: float = 0.1,
                 n_refine: int = 10, n_samples: int = 12, use_infonce: bool = True,
                 device: str = "cuda"):
        super().__init__()
        self.beta = beta
        self.m = m
        self.lr_refine = lr_refine
        self.n_refine = n_refine
        self.n_samples = n_samples
        self.use_infonce = use_infonce
        self.device = device if torch.cuda.is_available() and "cuda" in device else ("cpu" if device == "cpu" else torch.device(device))

    def forward(self,
                fg_proto: torch.Tensor,      # [K_fg, C] (Assumed Normalized)
                bg_proto: torch.Tensor,      # [K_bg, C] (Assumed Normalized)
                F_map: torch.Tensor,         # [N, C, H, W]
                M_map: torch.Tensor          # [N, 1, H, W]
                ):
        """
        Returns: (loss_scalar, refined_fg_proto, refined_bg_proto)
        """
        if self.n_refine == 0:
            zero = torch.tensor(0.0, device=fg_proto.device, dtype=fg_proto.dtype)
            return zero, fg_proto, bg_proto

        if M_map.ndim == 3:
            M_map = M_map.unsqueeze(1)
        N, C, H, W = F_map.shape

        # Resize masks to feature resolution
        M_resized = F.interpolate(M_map.float(), size=(H, W), mode='nearest').clamp(0, 1)

        # Flatten
        F_flat = F_map.permute(0, 2, 3, 1).reshape(-1, C)    # [N*H*W, C]
        M_flat = M_resized.reshape(-1)                       # [N*H*W]

        # Normalize features ONLY (Prototypes are assumed pre-normalized by ProtoSegNet)
        F_flat = safe_norm(F_flat, dim=-1)
        # OPTIMIZATION: Removed redundant safe_norm on fg_proto/bg_proto here
        fg_proto_n = fg_proto.to(F_flat.device)
        bg_proto_n = bg_proto.to(F_flat.device)

        # Weights
        fg_weight = M_flat
        bg_weight = 1.0 - M_flat

        # Cosine distance: 1 - Sim
        dist_fg_all = 1.0 - (F_flat @ fg_proto_n.T)
        dist_bg_all = 1.0 - (F_flat @ bg_proto_n.T)

        dist_fg_min, _ = dist_fg_all.min(dim=1)
        dist_bg_min, _ = dist_bg_all.min(dim=1)

        # Select hard examples
        k_fg = min(self.n_samples, F_flat.shape[0])
        k_bg = min(self.n_samples, F_flat.shape[0])

        score_fg = (dist_fg_min * fg_weight).detach()
        score_bg = (dist_bg_min * bg_weight).detach()

        pos_idx = torch.topk(score_fg, k=k_fg, largest=True).indices
        neg_idx = torch.topk(score_bg, k=k_bg, largest=True).indices

        # Detached clones for iterative refinement
        p_fg = fg_proto_n.detach().clone().to(self.device)
        p_bg = bg_proto_n.detach().clone().to(self.device)

        pos_feats = F_flat[pos_idx].to(self.device)
        neg_feats = F_flat[neg_idx].to(self.device)

        # Refine Loop
        with torch.no_grad():
            for it in range(self.n_refine):
                step = float(self.lr_refine) / (1.0 + it * 0.5)

                if pos_feats.numel() > 0:
                    fg_sim = pos_feats @ p_fg.T
                    fg_assign = fg_sim.argmax(dim=1)
                    for k in range(p_fg.shape[0]):
                        mask_k = (fg_assign == k)
                        if mask_k.any():
                            cluster_mean = pos_feats[mask_k].mean(dim=0)
                            new_cent = (1.0 - step) * p_fg[k] + step * cluster_mean
                            p_fg[k] = safe_norm(new_cent.unsqueeze(0), dim=-1)[0]

                if neg_feats.numel() > 0:
                    bg_sim = neg_feats @ p_bg.T
                    bg_assign = bg_sim.argmax(dim=1)
                    for k in range(p_bg.shape[0]):
                        mask_k = (bg_assign == k)
                        if mask_k.any():
                            cluster_mean = neg_feats[mask_k].mean(dim=0)
                            new_cent = (1.0 - step) * p_bg[k] + step * cluster_mean
                            p_bg[k] = safe_norm(new_cent.unsqueeze(0), dim=-1)[0]

        # Compute losses (Contrastive)
        loss_triplet = self.triplet_loss(F_flat.to(self.device), pos_idx.to(self.device),
                                         neg_idx.to(self.device), p_fg, p_bg)
        loss = loss_triplet
        if self.use_infonce:
            loss = loss + 0.25 * self.infonce_loss(F_flat.to(self.device), pos_idx.to(self.device),
                                                   neg_idx.to(self.device), p_fg, p_bg)

        # Momentum update
        p_fg_back = p_fg.to(fg_proto.device).type_as(fg_proto)
        p_bg_back = p_bg.to(bg_proto.device).type_as(bg_proto)

        refined_fg = safe_norm((1.0 - self.m) * fg_proto + self.m * p_fg_back, dim=-1)
        refined_bg = safe_norm((1.0 - self.m) * bg_proto + self.m * p_bg_back, dim=-1)

        loss = loss.to(fg_proto.device)
        return loss, refined_fg, refined_bg

    def triplet_loss(self, F_flat, pos_idx, neg_idx, p_fg, p_bg, margin=0.2):
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            return torch.tensor(0.0, device=p_fg.device, dtype=p_fg.dtype)
        pos_feats = safe_norm(F_flat[pos_idx], dim=-1)
        neg_feats = safe_norm(F_flat[neg_idx], dim=-1)
        sim_pos = (pos_feats @ p_fg.T).max(dim=1)[0].mean()
        sim_neg = (neg_feats @ p_fg.T).max(dim=1)[0].mean()
        return F.relu(margin + sim_neg - sim_pos)

    def infonce_loss(self, F_flat, pos_idx, neg_idx, p_fg, p_bg, tau=0.07):
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            return torch.tensor(0.0, device=p_fg.device)

        pos_feats = safe_norm(F_flat[pos_idx], dim=-1)
        neg_feats = safe_norm(F_flat[neg_idx], dim=-1)

        # Calculate similarity to ALL fg prototypes
        # [N_pos, K_fg]
        pos_sim_matrix = (pos_feats @ p_fg.T) / tau

        # Standard InfoNCE logic: LogSumExp
        # We want to maximize similarity to the "closest" prototype,
        # or treat all prototypes as valid positive keys.

        # LogSumExp trick for numerical stability
        # Numerator: Sum of exp(sim) for all FG prototypes
        numerator = torch.logsumexp(pos_sim_matrix, dim=1)

        # For denominator, we need negative samples.
        # Usually InfoNCE contrasts a query against 1 positive key and K negative keys.
        # Here, you are treating bg_prototypes as negative keys.
        neg_sim_matrix = (pos_feats @ p_bg.T) / tau

        # Denominator: Sum of (exp(pos) + exp(neg))
        all_sims = torch.cat([pos_sim_matrix, neg_sim_matrix], dim=1)
        denominator = torch.logsumexp(all_sims, dim=1)

        return -(numerator - denominator).mean()

# -------------------------
# ProtoSegNet (Refined)
# -------------------------
class ProtoSegNet(nn.Module):
    def __init__(self, encoder, output_size=256, fusion='softmax', use_refiner=True):
        super().__init__()
        self.encoder = encoder
        self.output_size = output_size
        self.fusion = fusion
        self.temperature = 20.0
        self.use_refiner = use_refiner

        if self.use_refiner:
            self.refiner = PrototypeRefiner()
        else:
            self.refiner = None

    def get_prototype(self, feat, mask, mode='community', thresh=0.95, n_clusters=10):
        """
        Generates Normalized Prototypes.
        """
        N, C, Hf, Wf = feat.shape
        mask = F.interpolate(mask.float(), size=(Hf, Wf), mode='nearest')

        if mode == 'mask':
            proto_list = []
            for i in range(N):
                proto = (feat[i:i+1] * mask[i:i+1]).sum(dim=(2, 3)) / (mask[i:i+1].sum(dim=(2, 3)) + 1e-5)
                proto = safe_norm(proto, dim=1)
                proto_list.append((proto, proto))
            return proto_list

        elif mode == 'gridconv+':
            grid_size = int(round(16 ** 0.5))
            k_h, k_w = max(1, Hf // grid_size), max(1, Wf // grid_size)
            feat_grid = F.avg_pool2d(feat, (k_h, k_w), (k_h, k_w))
            mask_grid = F.avg_pool2d(mask, (k_h, k_w), (k_h, k_w))
            feat_flat = feat_grid.flatten(2).permute(0, 2, 1)
            mask_flat = mask_grid.flatten(2)

            proto_list = []
            for i in range(N):
                fg_mask = mask_flat[i, 0] > thresh
                bg_mask = mask_flat[i, 0] <= (1 - thresh)
                fg_feats = feat_flat[i][fg_mask] if fg_mask.any() else torch.zeros(0, C, device=feat.device)
                bg_feats = feat_flat[i][bg_mask] if bg_mask.any() else torch.zeros(0, C, device=feat.device)

                fg_global = (feat[i:i+1] * mask[i:i+1]).sum(dim=(2, 3)) / (mask[i:i+1].sum(dim=(2, 3)) + 1e-5)
                bg_global = (feat[i:i+1] * (1 - mask[i:i+1])).sum(dim=(2, 3)) / ((1 - mask[i:i+1]).sum(dim=(2, 3)) + 1e-5)

                all_fg = torch.cat([fg_global, fg_feats], dim=0) if fg_feats.numel() else fg_global
                all_bg = torch.cat([bg_global, bg_feats], dim=0) if bg_feats.numel() else bg_global

                proto_list.append((safe_norm(all_fg, dim=1), safe_norm(all_bg, dim=1)))
            return proto_list

        elif mode == 'community':
            proto_list = []
            for i in range(N):
                fg_mask = mask[i, 0] > thresh
                bg_mask = mask[i, 0] < (1 - thresh)

                fg_feats = feat[i, :, fg_mask].T if fg_mask.any() else torch.zeros(0, C, device=feat.device)
                bg_feats = feat[i, :, bg_mask].T if bg_mask.any() else torch.zeros(0, C, device=feat.device)

                fg_local = kmeans_torch(fg_feats, n_clusters=n_clusters, n_iters=20, device=feat.device)
                bg_local = kmeans_torch(bg_feats, n_clusters=n_clusters, n_iters=20, device=feat.device)

                def mean_proto(feats):
                    if feats.numel() == 0:
                        return torch.zeros(1, C, device=feats.device)

                    proto = feats.mean(dim=0, keepdim=True)  # simple mean
                    return safe_norm(proto, dim=1)           # normalize

                fg_global = mean_proto(fg_feats)
                bg_global = mean_proto(bg_feats)

                fg_sample = torch.cat([fg_global, fg_local], dim=0) if fg_local.numel() else fg_global
                bg_sample = torch.cat([bg_global, bg_local], dim=0) if bg_local.numel() else bg_global

                proto_list.append((fg_sample, bg_sample))
            return proto_list

        else:
            raise NotImplementedError(f"Mode '{mode}' not supported")

    def get_prediction(self, query_feat, proto_list, return_logits=True):
        """
        Computes similarity (logits).
        FIX: Defaults to return_logits=True for better fusion logic.
        """
        B, C, H, W = query_feat.shape
        query_feat = safe_norm(query_feat, dim=1)
        temp = self.temperature

        def conv_sim_single(query_single, protos):
            if protos is None or protos.numel() == 0:
                return torch.zeros(1, 1, H, W, device=query_feat.device)
            weight = protos.unsqueeze(-1).unsqueeze(-1)
            sim = F.conv2d(query_single, weight)
            return temp * sim

        def fuse(sim):
            if sim.shape[1] == 1: return sim
            if self.fusion == 'mean': return sim.mean(dim=1, keepdim=True)
            if self.fusion == 'max': return sim.max(dim=1, keepdim=True)[0]
            if self.fusion == 'softmax':
                att = F.softmax(sim, dim=1)
                return (att * sim).sum(dim=1, keepdim=True)
            raise ValueError(f"Unknown fusion: {self.fusion}")

        results = []
        for i in range(B):
            fg_proto, bg_proto = proto_list[i]
            fg_sim = conv_sim_single(query_feat[i:i+1], fg_proto)
            bg_sim = conv_sim_single(query_feat[i:i+1], bg_proto)

            logits_fg = fuse(fg_sim)
            logits_bg = fuse(bg_sim)

            logits = torch.cat([logits_bg, logits_fg], dim=1) # [1, 2, H, W]

            if return_logits:
                results.append(logits)
            else:
                results.append(F.softmax(logits, dim=1))

        return torch.cat(results, dim=0)

    def logo_fusion(self, S_prime, S_tilde_prime):
        """
        LoGo Fusion using LOGITS.
        S_prime: Global Logits
        S_tilde_prime: Local Logits
        """
        # Handle bg/fg channels
        if S_prime.shape[1] == 2 and S_tilde_prime.shape[1] == 2:
            bg_fused = self._fuse_single_channel(S_prime[:, 0:1], S_tilde_prime[:, 0:1])
            fg_fused = self._fuse_single_channel(S_prime[:, 1:2], S_tilde_prime[:, 1:2])
            return torch.cat([bg_fused, fg_fused], dim=1)
        else:
            return self._fuse_single_channel(S_prime, S_tilde_prime)

    def _fuse_single_channel(self, S_prime, S_tilde_prime):
        """
        Fusion logic on LOGITS.
        Calculates gates via Softmax on the stacked logits.
        """
        stacked = torch.cat([S_prime, S_tilde_prime], dim=1) # [B, 2, H, W]

        # Softmax here determines which branch (Global vs Local) is more confident
        gates = F.softmax(20*stacked, dim=1)
        G = gates[:, 0:1, :, :]
        one_minus_G = gates[:, 1:2, :, :]

        # Weighted sum of LOGITS
        S_hat = G * S_prime + one_minus_G * S_tilde_prime + S_prime
        return S_hat

    def forward(self, support_img, support_mask, query_img, mode='community', n_clusters=9):
        """
        Forward pass returning LOGITS (sim_map) for stability.
        """
        B = support_img.shape[0]

        # Feature Extraction
        support_feat = self.encoder(support_img)
        if isinstance(support_feat, tuple): support_feat = support_feat[2]

        query_feat = self.encoder(query_img)
        if isinstance(query_feat, tuple): query_feat = query_feat[2]

        # 1. Generate Initial Prototypes
        proto_list = self.get_prototype(support_feat, support_mask, mode=mode, n_clusters=n_clusters)

        # 2. Refine Prototypes
        if self.use_refiner:
            refined_proto_list = []
            triplet_losses = []
            for i in range(B):
                fg_proto, bg_proto = proto_list[i]
                # Optimization: proto_list is already normalized, so Refiner skips it
                triplet_loss, refined_fg, refined_bg = self.refiner(
                    fg_proto, bg_proto,
                    support_feat[i:i+1], support_mask[i:i+1]
                )
                refined_proto_list.append((refined_fg, refined_bg))
                triplet_losses.append(triplet_loss)
            triplet_loss = torch.stack(triplet_losses).mean()
        else:
            refined_proto_list = proto_list
            triplet_loss = torch.tensor(0.0, device=query_feat.device)

        # 3. Get Logits (Not Probabilities)

        # Pixel-level (Global/Refined)
        pixel_level_logits = self.get_prediction(query_feat, refined_proto_list, return_logits=False)

        # Local-level (Context)
        query_feat_context = F.avg_pool2d(query_feat, 3, 1, 1)
        sup_feat_context = F.avg_pool2d(support_feat, 3, 1, 1)

        local_proto_list = self.get_prototype(sup_feat_context, support_mask, mode='community', n_clusters=n_clusters)
        local_level_logits = self.get_prediction(query_feat_context, local_proto_list, return_logits=False)

        # 4. LoGo Fusion (on Logits)
        sim_map_logits = self.logo_fusion(pixel_level_logits, local_level_logits)

        # 5. Reversed Map (for consistency check, also Logits)
        reversed_map_logits = self.get_prediction(support_feat, refined_proto_list, return_logits=True)

        #add softmax here before interpolate
        sim_map_logits = F.softmax(sim_map_logits, dim=1)

        reversed_map_logits = F.softmax(reversed_map_logits, dim=1)
        # 6. Interpolate Logits

        sim_map = F.interpolate(sim_map_logits, size=(self.output_size, self.output_size),
                                mode='bilinear', align_corners=True)
        reversed_map = F.interpolate(reversed_map_logits, size=(self.output_size, self.output_size),
                                     mode='bilinear', align_corners=True)

        return {
            'sim_map': sim_map,               # [B, 2, H, W] LOGITS (Use CE Loss directly)
            'reversed_map': reversed_map,     # [B, 2, H, W] LOGITS
            'triplet_loss': triplet_loss,
            'query_feat': query_feat,
            'support_feat': support_feat,
            'proto_list': proto_list,
            'refined_proto_list': refined_proto_list
        }
