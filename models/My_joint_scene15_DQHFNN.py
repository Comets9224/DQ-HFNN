"""
Scene15 Dual-Qubit Joint Membership Model - Hierarchical Dynamic Sampling Strategy (Interface-Compatible Version)
- Classical DNN + Dual-Qubit Joint Membership
- Hierarchical Sampling: Local + Mid-range + Global
- Fully Vectorized, GPU-Friendly
- Strictly Aligns with QAHFNN and FDNN Classical Branches
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torchquantum as tq
import numpy as np

# ==================== Classical Feature Extractor (Identical to QAHFNN/FDNN) ====================
class DenseNet(nn.Module):
    """
    Scene15 Classical Feature Extractor (Strictly Reproducing Author's Implementation)
    3-layer fully connected network for extracting high-dimensional features from 200D SIFT features

    Key Details:
    - Dropout applied **before** each linear layer (including first layer)
    - **No** Dropout after final layer output
    - Uses 40% Dropout probability
    """

    def __init__(self, input_dim=200, hidden_dim=256):
        super(DenseNet, self).__init__()

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.4)
        self.acti = nn.ReLU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.acti(self.l1(x))
        x = self.dropout(x)
        x = self.acti(self.l2(x))
        x = self.dropout(x)
        x = self.acti(self.l3(x))
        return x


# ==================== Dual-Qubit Joint Membership Circuit ====================
class JointMembership_QuantumCircuit(nn.Module):
    """
    Dual-Qubit Joint Membership Quantum Circuit

    Circuit Structure:
    q0: Encode → RZ → CNOT(control) → RY → CNOT(control) → Measure ⟨Z₀⟩ → μ₀(x₁,x₂)
                      ↓                  ↓
    q1: Encode ───── CNOT(target) ────── CNOT(target) ─→ Measure ⟨Z₁⟩ → μ₁(x₁,x₂)

    Output: [batch, 2] - Joint membership from both qubits
    """

    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 2


            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)


            self.cnot1 = tq.CNOT()
            self.cnot2 = tq.CNOT()

        def forward(self, device: tq.QuantumDevice):

            self.rz0(device, wires=0)
            self.cnot1(device, wires=[0, 1])
            self.ry0(device, wires=0)
            self.cnot2(device, wires=[0, 1])

    def __init__(self):
        super().__init__()
        self.n_wires = 2
        self.encoder = tq.GeneralEncoder([
            {'input_idx': [0], 'func': 'ry', 'wires': [0]},
            {'input_idx': [1], 'func': 'ry', 'wires': [1]},
        ])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):

        bsz = x.shape[0]
        assert x.shape[1] == 2, f"期望输入维度为2，实际为{x.shape[1]}"
        device = tq.QuantumDevice(n_wires=2, bsz=bsz, device=x.device)


        self.encoder(device, x)
        self.q_layer(device)


        measurements = self.measure(device)  # [batch, 2]
        qout = (measurements + 1) / 2
        return qout


# ==================== 分层动态随机采样模糊层 ====================
class StratifiedSampling_FuzzyLayer_Scene15(nn.Module):
    """
    Scene15-Specific Hierarchical Dynamic Sampling Fuzzy Layer

    Core Concept:
    No predefined feature importance, but explores the "distance" dimension between features.
    Divides all possible pairing relationships into three levels and dynamically samples from each:

    1. **Local Pairs**:
       - Captures relationships between nearby dimensions
       - Randomly selects pairs within window radius
       - Example: Feature i paired with feature (i±W)

    2. **Mid-range Pairs**:
       - Captures relationships between adjacent feature blocks
       - Divides 200D into blocks, pairs across adjacent blocks
       - Example: Feature from Block 1 paired with feature from Block 2

    3. **Global Pairs**:
       - Explores relationships between any two dimensions
       - Completely random pairing, no positional constraints
       - Example: Feature 0 paired with feature 199

    Parameters:
    - num_classes: Number of classes (Scene15=15)
    - n_pairs: Total pairs (default=60)
    - n_local: Local pairs (default=20, passed via n_top_bottom_pairs)
    - n_mid: Mid-range pairs (default=20, passed via n_top_median_pairs)
    - n_global: Global pairs (auto-calculated = n_pairs - n_local - n_mid)
    - window_size: Local window radius (default=5, passed via top_k)
    - num_blocks: Mid-range block count (default=4, passed via bottom_m)

    ⚠️ Parameter Mapping (for interface compatibility):
    - n_top_bottom_pairs → n_local (local pairs)
    - n_top_median_pairs → n_mid (mid-range pairs)
    - top_k → window_size (local window radius)
    - bottom_m → num_blocks (mid-range block count)
    - median_range → Reserved but unused (compatibility)

    Input: [batch, 200] SIFT histogram features
    Output: [batch, n_pairs*2, num_classes] = [batch, 120, 15] (assuming n_pairs=60)
    """
    def __init__(self, num_classes=15, n_pairs=60, n_top_bottom_pairs=20,
                 n_top_median_pairs=20, top_k=5, bottom_m=4,
                 median_range=(80, 120)):
        super(StratifiedSampling_FuzzyLayer_Scene15, self).__init__()
        self.num_classes = num_classes
        self.n_pairs = n_pairs


        self.n_local = n_top_bottom_pairs
        self.n_mid = n_top_median_pairs
        self.window_size = top_k
        self.num_blocks = bottom_m
        self.median_range = median_range


        self.n_global = n_pairs - self.n_local - self.n_mid


        assert self.n_local + self.n_mid + self.n_global == n_pairs, \
            f"Local({self.n_local}) + Mid-range({self.n_mid}) + Global({self.n_global}) must equal total pairs({n_pairs})"
        assert self.n_local >= 0 and self.n_mid >= 0 and self.n_global >= 0, \
            "All pair counts must be non-negative"
        assert self.window_size >= 1 and self.window_size < 100, \
            f"Window radius({self.window_size}) must be in [1, 100) range"
        assert self.num_blocks >= 2 and self.num_blocks <= 200, \
            f"Block count({self.num_blocks}) must be in [2, 200] range"
        assert 200 % self.num_blocks == 0, \
            f"200 dimensions must be divisible by block count({self.num_blocks})"
        self.block_size = 200 // self.num_blocks
        self.qfuzziers = nn.ModuleList([
            JointMembership_QuantumCircuit() for _ in range(num_classes)
        ])

        print(f"\n{'=' * 70}")
        print(f"【Scene15 Hierarchical Dynamic Sampling Fuzzy Layer】")
        print(f"{'=' * 70}")
        print(f"  Core Concept:")
        print(f"    No predefined feature importance, explores multi-scale feature relationships")
        print(f"\n  Sampling Strategy:")
        print(f"    - Total Pairs: {n_pairs} pairs")
        print(f"    - Local Pairs: {self.n_local} pairs ({self.n_local / n_pairs * 100:.1f}%)")
        print(f"        └─ Window Radius: ±{self.window_size} (covers {2 * self.window_size + 1} dims)")
        print(f"    - Mid-range Pairs: {self.n_mid} pairs ({self.n_mid / n_pairs * 100:.1f}%)")
        print(f"        └─ Blocks: {self.num_blocks} ({self.block_size} dims/block)")
        print(f"    - Global Pairs: {self.n_global} pairs ({self.n_global / n_pairs * 100:.1f}%)")
        print(f"        └─ Range: Any two dimensions (0-199)")
        print(f"\n  Sampling Features:")
        print(f"    ✅ Fully Dynamic: Different pairs per batch")
        print(f"    ✅ Fully Vectorized: GPU-friendly, no Python loops")
        print(f"    ✅ Multi-scale Exploration: Local + Block-wise + Global patterns")
        print(f"\n  Quantum Computation:")
        print(f"    - Quantum Circuits: {num_classes} (independent per class)")
        print(f"    - Qubits: 2")
        print(f"    - Circuit Calls/Sample: {n_pairs * num_classes}")
        print(f"    - Output Dimension: [batch, {n_pairs * 2}, {num_classes}]")
        print(f"\n  Parameter Mapping (Interface Compatible):")
        print(f"    - n_top_bottom_pairs → n_local")
        print(f"    - n_top_median_pairs → n_mid")
        print(f"    - top_k → window_size")
        print(f"    - bottom_m → num_blocks")
        print(f"{'=' * 70}\n")
    def _generate_local_pairs(self, batch_size, device):
        """
        Generate local random pairs

        Strategy:
        1. Randomly select n_local center points i (avoiding boundaries)
        2. Randomly select offset within window [-W, W] for each center
        3. Ensure offset ≠ 0 (avoid self-pairing)
        4. Calculate paired point j = i + offset

        Parameters:
            batch_size: Batch size
            device: Target device

        Returns:
            [batch, n_local, 2] pair indices
        """
        if self.n_local == 0:
            return torch.empty(batch_size, 0, 2, dtype=torch.long, device=device)


        center_indices = torch.randint(
            low=self.window_size,
            high=200 - self.window_size,
            size=(batch_size, self.n_local),
            device=device
        )  # [batch, n_local]


        offsets = torch.randint(
            low=0,
            high=2 * self.window_size + 1,
            size=(batch_size, self.n_local),
            device=device
        ) - self.window_size  # [batch, n_local] 范围：[-W, W]


        offsets[offsets == 0] = 1


        pair_indices = center_indices + offsets  # [batch, n_local]


        local_pairs = torch.stack([center_indices, pair_indices], dim=-1)  # [batch, n_local, 2]

        return local_pairs

    def _generate_mid_range_pairs(self, batch_size, device):
        """
        Generate mid-range random pairs

        Strategy:
        1. Randomly select n_mid starting blocks (excluding last block)
        2. Randomly select a point i within starting block
        3. Randomly select a point j in adjacent next block
        4. Form pair (i, j)

        Parameters:
            batch_size: Batch size
            device: Target device

        Returns:
            [batch, n_mid, 2] pair indices
        """
        if self.n_mid == 0:
            return torch.empty(batch_size, 0, 2, dtype=torch.long, device=device)


        start_blocks = torch.randint(
            low=0,
            high=self.num_blocks - 1,
            size=(batch_size, self.n_mid),
            device=device
        )  # [batch, n_mid]


        offsets1 = torch.randint(
            0, self.block_size,
            size=(batch_size, self.n_mid),
            device=device
        )
        indices1 = start_blocks * self.block_size + offsets1  # [batch, n_mid]


        offsets2 = torch.randint(
            0, self.block_size,
            size=(batch_size, self.n_mid),
            device=device
        )
        indices2 = (start_blocks + 1) * self.block_size + offsets2  # [batch, n_mid]


        mid_range_pairs = torch.stack([indices1, indices2], dim=-1)  # [batch, n_mid, 2]

        return mid_range_pairs

    def _generate_global_pairs(self, batch_size, device):

        if self.n_global == 0:
            return torch.empty(batch_size, 0, 2, dtype=torch.long, device=device)


        indices1 = torch.randint(0, 200, size=(batch_size, self.n_global), device=device)
        indices2 = torch.randint(0, 200, size=(batch_size, self.n_global), device=device)


        mask = indices1 == indices2
        indices2[mask] = (indices2[mask] + 1) % 200  # 加1并取模，避免越界


        global_pairs = torch.stack([indices1, indices2], dim=-1)  # [batch, n_global, 2]

        return global_pairs

    def forward(self, x):

        batch_size = x.size(0)
        device = x.device


        local_pairs = self._generate_local_pairs(batch_size, device)  # [batch, n_local, 2]
        mid_pairs = self._generate_mid_range_pairs(batch_size, device)  # [batch, n_mid, 2]
        global_pairs = self._generate_global_pairs(batch_size, device)  # [batch, n_global, 2]


        all_pairs_idx = torch.cat([local_pairs, mid_pairs, global_pairs], dim=1)  # [batch, n_pairs, 2]


        x_expanded = x.unsqueeze(1).expand(-1, self.n_pairs, -1)  # [batch, n_pairs, 200]


        idx_0 = all_pairs_idx[:, :, 0].unsqueeze(-1)  # [batch, n_pairs, 1]
        idx_1 = all_pairs_idx[:, :, 1].unsqueeze(-1)  # [batch, n_pairs, 1]


        val_0 = torch.gather(x_expanded, 2, idx_0).squeeze(-1)  # [batch, n_pairs]
        val_1 = torch.gather(x_expanded, 2, idx_1).squeeze(-1)  # [batch, n_pairs]


        pairs_values = torch.stack([val_0, val_1], dim=-1)  # [batch, n_pairs, 2]


        outputs = []
        for qfuzzier in self.qfuzziers:
            pairs_flat = pairs_values.view(-1, 2)  # [batch*n_pairs, 2]
            qout = qfuzzier(pairs_flat)  # [batch*n_pairs, 2]
            qout = qout.view(batch_size, self.n_pairs, 2)  # [batch, n_pairs, 2]
            class_output = qout.view(batch_size, -1)  # [batch, n_pairs*2]
            outputs.append(class_output)

        return torch.stack(outputs, dim=-1)  # [batch, n_pairs*2, num_classes]


# ==================== Complete Model ====================
class SCENE15_JointMembership(nn.Module):
    """
    Scene15 Dual-Qubit Joint Membership Hybrid Network - Hierarchical Dynamic Sampling Strategy

    Architecture:
        1. Classical Branch: 3-layer FC feature extractor (identical to QAHFNN/FDNN)
        2. Dual-Qubit Branch: Hierarchical dynamic sampling strategy
        3. Fusion Layer: ADD fusion
        4. Classifier: 4-layer FC (identical to QAHFNN/FDNN)

    Parameter Notes (maintaining full interface compatibility):
        num_classes: Number of classes (Scene15=15)
        hidden_dim: Hidden dimension (default 256)
        n_qnn_layers: Reserved parameter (compatibility, unused)
        n_random_pairs: Reserved parameter (compatibility, maps to n_local)
        n_pairs: Total pairs (default 60)
        n_top_bottom_pairs: Local pairs (default 20, or uses n_random_pairs)
        n_top_median_pairs: Mid-range pairs (default 20)
        top_k: Local window radius (default 5)
        bottom_m: Mid-range blocks (default 4)
        median_range: Reserved but unused (compatibility)
        pair_ratio: Pair ratio (optional, tuple/list format (local_ratio, mid_ratio))

    ⚠️ Parameter Mapping:
        - n_top_bottom_pairs / n_random_pairs → n_local (local pair count)
        - n_top_median_pairs → n_mid (mid-range pair count)
        - top_k → window_size (local window radius)
        - bottom_m → num_blocks (mid-range block count)
    """
    def __init__(self, num_classes=15, hidden_dim=256, n_qnn_layers=1,
                 n_random_pairs=20, n_pairs=None, n_top_bottom_pairs=None,
                 n_top_median_pairs=None, top_k=5, bottom_m=4,
                 median_range=(80, 120), pair_ratio=None):
        super(SCENE15_JointMembership, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim


        if isinstance(median_range, list):
            median_range = tuple(median_range)


        if pair_ratio is not None and isinstance(pair_ratio, list):
            pair_ratio = tuple(pair_ratio)


        if n_pairs is None:
            n_pairs = 60


        if pair_ratio is not None:
            ratio_local, ratio_mid = pair_ratio
            if ratio_local + ratio_mid > 1.0 + 1e-6:
                raise ValueError(
                    f"Sum of pair_ratio values cannot exceed 1.0, "
                    f"actual value is {ratio_local + ratio_mid:.4f}. "
                    f"Remaining portion ({1.0 - ratio_local - ratio_mid:.2%}) will be automatically allocated to global pairs."
                )
            n_local = int(n_pairs * ratio_local)
            n_mid = int(n_pairs * ratio_mid)
            n_global = n_pairs - n_local - n_mid

            print(f"\n✅ Using custom pair ratio:")
            print(f"   Local Pairs: {ratio_local * 100:.1f}% ({n_local} pairs)")
            print(f"   Mid-range Pairs: {ratio_mid * 100:.1f}% ({n_mid} pairs)")
            print(f"   Global Pairs: {(1 - ratio_local - ratio_mid) * 100:.1f}% ({n_global} pairs)\n")
            n_top_bottom_pairs = n_local
            n_top_median_pairs = n_mid

        elif n_top_bottom_pairs is None and n_top_median_pairs is None:

            n_local = n_pairs // 3
            n_mid = n_pairs // 3
            n_global = n_pairs - n_local - n_mid

            print(f"\n✅ Using default balanced pair ratio:")
            print(f"   Local Pairs: ~33% ({n_local} pairs)")
            print(f"   Mid-range Pairs: ~33% ({n_mid} pairs)")
            print(f"   Global Pairs: ~33% ({n_global} pairs)\n")

            n_top_bottom_pairs = n_local
            n_top_median_pairs = n_mid

        elif n_top_bottom_pairs is not None and n_top_median_pairs is not None:
            # 用户同时指定了两个数值
            n_local = n_top_bottom_pairs
            n_mid = n_top_median_pairs
            n_global = n_pairs - n_local - n_mid

            if n_global < 0:
                raise ValueError(
                    f"n_top_bottom_pairs ({n_local}) + "
                    f"n_top_median_pairs ({n_mid}) "
                    f"cannot exceed n_pairs ({n_pairs})"
                )

            print(f"\n✅ Using manually specified pair counts:")
            print(f"   Local Pairs: {n_local} pairs ({n_local / n_pairs * 100:.1f}%)")
            print(f"   Mid-range Pairs: {n_mid} pairs ({n_mid / n_pairs * 100:.1f}%)")
            print(f"   Global Pairs: {n_global} pairs ({n_global / n_pairs * 100:.1f}%)\n")
        else:
            # 只指定了一个，自动均分剩余配对
            if n_top_bottom_pairs is not None:
                n_local = n_top_bottom_pairs
                remaining = n_pairs - n_local
                n_mid = remaining // 2
                n_global = remaining - n_mid
                n_top_median_pairs = n_mid
            else:
                n_mid = n_top_median_pairs
                remaining = n_pairs - n_mid
                n_local = remaining // 2
                n_global = remaining - n_local
                n_top_bottom_pairs = n_local

            print(f"\n✅ Partially specified pair counts, automatically allocating remainder:")
            print(f"   Local Pairs: {n_local} pairs ({n_local / n_pairs * 100:.1f}%)")
            print(f"   Mid-range Pairs: {n_mid} pairs ({n_mid / n_pairs * 100:.1f}%)")
            print(f"   Global Pairs: {n_global} pairs ({n_global / n_pairs * 100:.1f}%)\n")
        # 1. 经典特征提取器（与QAHFNN/FDNN完全一致）
        self.class_layer = DenseNet(input_dim=200, hidden_dim=hidden_dim)

        # 2. 分层动态随机采样模糊分支
        self.quantum_fuzzy = StratifiedSampling_FuzzyLayer_Scene15(
            num_classes=num_classes,
            n_pairs=n_pairs,
            n_top_bottom_pairs=n_top_bottom_pairs,  # 映射为 n_local
            n_top_median_pairs=n_top_median_pairs,  # 映射为 n_mid
            top_k=top_k,  # 映射为 window_size
            bottom_m=bottom_m,  # 映射为 num_blocks
            median_range=median_range  # 保留但未使用
        )

        # 3. 融合层
        self.flinear = nn.Linear(num_classes, hidden_dim)

        # 4. 分类器（与QAHFNN/FDNN完全一致）
        self.classifier = nn.Sequential(
            nn.Dropout(0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        # 参数初始化（与QAHFNN/FDNN完全一致）
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                init.uniform_(m.weight,
                              a=-1 / torch.sqrt(torch.tensor(num_classes * 2, dtype=torch.float32)),
                              b=1 / torch.sqrt(torch.tensor(num_classes * 2, dtype=torch.float32)))
                if m.bias is not None:
                    init.zeros_(m.bias)

        print(f"\n{'=' * 70}")
        print(f"【Scene15 Hierarchical Dynamic Sampling Hybrid Network】")
        print(f"{'=' * 70}")
        print(f"  Architecture Components:")
        print(f"    - Classical Branch: 3-layer FC (identical to QAHFNN/FDNN)")
        print(f"    - Dual-Qubit Branch: Hierarchical dynamic sampling strategy")
        print(f"    - Fusion Strategy: ADD fusion")
        print(f"\n  Input Processing:")
        print(f"    - Input Size: [batch, 200] SIFT features")
        print(f"    - Hidden Dimension: {hidden_dim}")
        print(f"\n  Sampling Configuration:")
        print(f"    - Total Pairs: {n_pairs} pairs")
        print(f"    - Local Pairs: {n_top_bottom_pairs} pairs (window radius ±{top_k})")
        print(f"    - Mid-range Pairs: {n_top_median_pairs} pairs ({bottom_m} blocks)")
        print(f"    - Global Pairs: {n_pairs - n_top_bottom_pairs - n_top_median_pairs} pairs (fully random)")
        print(f"\n  Dimension Flow:")
        print(f"    - Quantum Output: [batch, {n_pairs * 2}, {num_classes}]")
        print(f"    - Fuzzy Rules: [batch, {num_classes}]")
        print(f"    - Fusion Dimension: {hidden_dim}")
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        classical_params = sum(p.numel() for p in self.class_layer.parameters() if p.requires_grad)
        quantum_params = sum(p.numel() for p in self.quantum_fuzzy.parameters() if p.requires_grad)
        print(f"\n  Model Scale:")
        print(f"    - Total Parameters: {total_params:,}")
        print(f"    - Classical Branch: {classical_params:,} ({classical_params / total_params * 100:.2f}%)")
        print(f"    - Quantum Branch: {quantum_params:,} ({quantum_params / total_params * 100:.2f}%)")

        print(f"{'=' * 70}\n")
    def forward(self, x):
        batch_size = x.size(0)
        c_part = self.class_layer(x)  # [batch, hidden_dim]
        fuzzy_memberships = self.quantum_fuzzy(x)  # [batch, n_pairs*2, num_classes]
        fuzzy_memberships_clamped = torch.clamp(fuzzy_memberships, min=1e-7, max=1.0)
        log_membership = torch.log(fuzzy_memberships_clamped)
        fuzzy_features = torch.mean(log_membership, dim=1)  # [batch, num_classes]
        fuzzy_features = self.flinear(fuzzy_features)  # [batch, hidden_dim]
        fusion_output = torch.add(c_part, fuzzy_features)  # [batch, hidden_dim]
        x = self.classifier(fusion_output)  # [batch, num_classes]
        return x
    def get_fused_features(self, x):

        batch_size = x.size(0)
        c_part = self.class_layer(x)
        fuzzy_memberships = self.quantum_fuzzy(x)
        fuzzy_memberships_clamped = torch.clamp(fuzzy_memberships, min=1e-7, max=1.0)
        log_membership = torch.log(fuzzy_memberships_clamped)
        fuzzy_features = torch.mean(log_membership, dim=1)
        fuzzy_features = self.flinear(fuzzy_features)
        return torch.add(c_part, fuzzy_features)

    def get_classical_features(self, x):

        return self.class_layer(x)

    def get_quantum_features(self, x):

        fuzzy_memberships = self.quantum_fuzzy(x)
        fuzzy_memberships_clamped = torch.clamp(fuzzy_memberships, min=1e-7, max=1.0)
        log_membership = torch.log(fuzzy_memberships_clamped)
        fuzzy_features = torch.mean(log_membership, dim=1)
        return self.flinear(fuzzy_features)

