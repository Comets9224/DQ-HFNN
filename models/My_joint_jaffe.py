"""
JAFFE Dual-Qubit Joint Membership Hybrid Neural Network (No-Pooling Modified Version with Batch Parallel Optimization + Mutual Information Analysis)

✅ New Features (2025-01-18):
1. _get_fixed_features() - Extracts fixed pair features separately (for mutual information analysis)
2. _get_random_features() - Extracts random pair features separately (for mutual information analysis)
3. Supports runtime dynamic adjustment of random pair ratio (no retraining needed)

Existing Features:
1. Precomputed valid pixel indices (avoids runtime filtering)
2. Batch-parallel random sampling (eliminates nested loops)
3. Vectorized index operations (GPU-accelerated)
4. Precomputed sampling plan (avoids runtime calculations)

Architecture Components:
1. Classical CNN: 6 conv layers + 2 residual blocks (from paper reproduction)
2. Quantum Fuzzy: Dual-qubit joint membership + 3x3 grid partition (no-pooling optimized)
3. Fusion Layer: ADD fusion
4. Classifier: Fully-connected layer

Modified Configuration (2025-01-19):
- Total pairs: 256 → 512
- Fixed pairs: 179 → 358 (~70%)
- Random pairs: 77 → 154 (~30%, default)
- Output dimension: [batch, 512, 7] → [batch, 1024, 7]
"""
import torch
import torch.nn as nn
import torchquantum as tq


# ================== Quantum Circuit: Joint Membership-Single Layer-With Entanglement (Reused from DMNIST) =======================
class JointMembership_QuantumCircuit(nn.Module):
    """
    Joint Membership Quantum Circuit Model (Reused from DMNIST version)
    - 2 qubits
    - 1 layer (single data encoding)
    - Contains entanglement gate (CNOT)
    - **Simultaneous measurement of both qubits**, outputs 2D joint membership

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

            # Instantiate all trainable rotation gates in __init__ (maintaining similar parameterization as original)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)

            self.cnot = tq.CNOT()

        def forward(self, device: tq.QuantumDevice):
            self.cnot(device, wires=[0, 1])
            self.ry0(device, wires=0)

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

        measurements = self.measure(device)  # shape: [batch, 2]
        qout = (measurements + 1) / 2
        return qout  # [batch, 2]


# ========= Quantum Fuzzy Layer: 3x3 Grid Partition Sampling (No-Pooling Modified Version - Batch Parallel Optimization + Mutual Information Analysis Interface) ===========
class JointMembership_FuzzyLayer_JAFFE_NoPool(nn.Module):
    """
    JAFFE-Specific Joint Membership Quantum Fuzzy Layer with 3x3 Grid Partition Sampling (No-Pooling Modified Version - Batch Parallel Optimization)

    ✅ Performance Optimizations:
    1. Precomputed valid pixel indices (_precompute_valid_pixels)
    2. Batch-parallel random sampling (_sample_inter_block_pairs_optimized)
    3. Vectorized index operations (eliminates inner loops)
    4. Precomputed sampling plan (_setup_random_sampling_plan)

    ✅ New Features
    5. _get_fixed_features() - Mutual information analysis interface
    6. _get_random_features() - Mutual information analysis interface
    7. Supports runtime dynamic ratio adjustment

    Modifications (vs original):
    1. **Total pairs**: 256 → 512
    2. **Grid boundaries**: [0,11,22,32] → [0,11,21,32] (more uniform)
    3. **Fixed pairs**: 179 → 358 (~70%)
    4. **Random pairs**: 77 → 154 (~30%, default)
    5. **Intra-block adjacent pairs**: ~602 → ~1204 (better spatial utilization)

    Grid Partition Details (32×32):
    - Block Boundaries: [0, 11, 21, 32]
    - Block Size Distribution:
      * Top-left/top-right/bottom-left/bottom-right: 11×11 = 121 pixels (4 blocks)
      * Top/bottom edges: 11×10 = 110 pixels (2 blocks)
      * Left/right edges: 10×11 = 110 pixels (2 blocks)
      * Center: 10×10 = 100 pixels (1 block)
    - Total Pixels: 4×121 + 2×110 + 2×110 + 100 = 1024✓

    Input: [batch, 1, 32, 32] - JAFFE grayscale images
    Output: [batch, 512*2, num_classes] = [batch, 1024, 7]

    Parameters:
    - num_classes: Number of classes (JAFFE=7)
    - n_random_pairs: Random pair count (default=154, ~30%)
    """

    def __init__(self, num_classes=7, n_random_pairs=154):
        super(JointMembership_FuzzyLayer_JAFFE_NoPool, self).__init__()
        self.num_classes = num_classes

        self.pooled_size = 32
        self.pixel_dim = 1024  # 32×32

        self.grid_size = 3
        self.block_sizes = [0, 11, 21, 32]

        self.total_pairs = 512
        self.n_random_pairs = n_random_pairs
        self.n_fixed_pairs = self.total_pairs - n_random_pairs

        assert 0 <= n_random_pairs <= 440, f"n_random_pairs={n_random_pairs} exceeds valid range [0, 440]"
        assert self.n_fixed_pairs >= 36, f"Insufficient fixed pairs (minimum 36, at least 4 per block required)"

        self.register_buffer('block_pixels', self._compute_block_partition())
        all_intra_pairs = self._compute_all_intra_pairs()
        self.register_buffer('intra_pair_indices',
                             self._sample_fixed_pairs(all_intra_pairs, self.n_fixed_pairs))
        self.register_buffer('inter_block_pairs', self._compute_inter_block_pairs())
        self.valid_pixels_per_block = self._precompute_valid_pixels()
        self.sampling_plan = self._setup_random_sampling_plan()
        self.qfuzziers = nn.ModuleList([
            JointMembership_QuantumCircuit() for _ in range(num_classes)
        ])
        print(f"\n{'=' * 70}")
        print(f"【JAFFE 3x3 Grid Quantum Fuzzy Layer (No-Pooling Modified - Dual-Qubit - Batch Parallel Optimized)】")
        print(f"{'=' * 70}")
        print(f"  Pooling Strategy: No pooling (direct raw image processing)")
        print(f"  Input Size: 32×32 = 1024 pixels")
        print(f"\n  Grid Configuration:")
        print(f"    - Partitions: 3×3 = 9 blocks")
        print(f"    - Block Boundaries: {self.block_sizes} (modified: more uniform distribution)")
        self._print_block_info()
        print(f"\n  Pairing Strategy (512-pair version):")
        print(f"    - Fixed Pairs: {self.n_fixed_pairs} pairs (intra-block 2D spatial adjacency)")
        print(f"    - Random Pairs: {self.n_random_pairs} pairs (cross-block dynamic sampling)")
        print(f"    - Total Pairs: {self.total_pairs} pairs")
        print(
            f"    - Pair Ratio: {self.n_fixed_pairs / self.total_pairs * 100:.1f}% fixed + {self.n_random_pairs / self.total_pairs * 100:.1f}% random")
        print(f"\n  Quantum Computation:")
        print(f"    - Quantum Circuits: {num_classes} (independent per class)")
        print(f"    - Circuit Calls/Sample: {self.total_pairs * num_classes}")
        print(f"    - Output Dimension: [batch, {self.total_pairs * 2}, {num_classes}]")
        print(f"\n  ✅ Performance Optimizations:")
        print(f"    - Precomputed valid pixel indices")
        print(f"    - Batch-parallel random sampling")
        print(f"    - Vectorized index operations")
        print(f"    - Precomputed sampling plan")
        print(f"{'=' * 70}\n")

    def _print_block_info(self):
        """Print 3x3 grid block details"""
        print(f"    - Block Size Distribution:")
        block_names = [
            "Top-left", "Top", "Top-right",
            "Left", "Center", "Right",
            "Bottom-left", "Bottom", "Bottom-right"
        ]
        total_pixels = 0
        for i in range(9):
            row_idx = i // 3
            col_idx = i % 3
            h = self.block_sizes[row_idx + 1] - self.block_sizes[row_idx]
            w = self.block_sizes[col_idx + 1] - self.block_sizes[col_idx]
            pixels = h * w
            total_pixels += pixels
            print(f"      * Block {i} ({block_names[i]:4s}): {h}×{w} = {pixels:3d} pixels")
        print(f"    - Total Pixel Verification: {total_pixels} (should be 1024)")

    def _compute_block_partition(self):
        """
        Precompute pixel indices for 3x3 grid partition (adapted for 32×32)

        Returns: [9, max_block_size] tensor
        """
        block_pixels_list = []
        max_size = 0

        for block_row in range(self.grid_size):
            for block_col in range(self.grid_size):
                pixels = []
                row_start = self.block_sizes[block_row]
                row_end = self.block_sizes[block_row + 1]
                col_start = self.block_sizes[block_col]
                col_end = self.block_sizes[block_col + 1]

                for r in range(row_start, row_end):
                    for c in range(col_start, col_end):
                        pixel_idx = r * self.pooled_size + c
                        pixels.append(pixel_idx)

                block_pixels_list.append(pixels)
                max_size = max(max_size, len(pixels))

        block_pixels_padded = []
        for pixels in block_pixels_list:
            padded = pixels + [-1] * (max_size - len(pixels))
            block_pixels_padded.append(padded)

        return torch.tensor(block_pixels_padded, dtype=torch.long)

    def _compute_all_intra_pairs(self):

        all_pairs = []

        for block_id in range(9):
            block_row = block_id // 3
            block_col = block_id % 3

            row_start = self.block_sizes[block_row]
            row_end = self.block_sizes[block_row + 1]
            col_start = self.block_sizes[block_col]
            col_end = self.block_sizes[block_col + 1]

            for r in range(row_start, row_end):
                for c in range(col_start, col_end - 1):
                    idx1 = r * self.pooled_size + c
                    idx2 = r * self.pooled_size + c + 1
                    all_pairs.append((idx1, idx2))

            for r in range(row_start, row_end - 1):
                for c in range(col_start, col_end):
                    idx1 = r * self.pooled_size + c
                    idx2 = (r + 1) * self.pooled_size + c
                    all_pairs.append((idx1, idx2))

        print(f"    - Intra-block adjacent pair pool: {len(all_pairs)} pairs (horizontal+vertical)")
        return all_pairs

    def _sample_fixed_pairs(self, all_pairs, n_fixed):
        """
        Uniformly sample fixed pairs from all possible pairs to ensure full image coverage

        Strategy: Use fixed random seed to ensure training reproducibility
        Parameters:
            all_pairs: List[Tuple] - All candidate pairs
            n_fixed: int - Number of fixed pairs to sample

        Returns: [n_fixed, 2] tensor
        """
        # Use fixed seed to ensure reproducibility
        rng = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(all_pairs), generator=rng)[:n_fixed]

        sampled_pairs = [all_pairs[i] for i in indices]
        covered_blocks = set()
        for p1, p2 in sampled_pairs:
            for pixel in [p1, p2]:
                row = pixel // self.pooled_size
                col = pixel % self.pooled_size

                block_row = 0 if row < 11 else (1 if row < 21 else 2)
                block_col = 0 if col < 11 else (1 if col < 21 else 2)
                covered_blocks.add(block_row * 3 + block_col)

        print(f"    - Fixed pairs cover blocks: {len(covered_blocks)}/9")
        return torch.tensor(sampled_pairs, dtype=torch.long)

    def _compute_inter_block_pairs(self):
        inter_pairs = []
        for row in range(3):
            for col in range(2):
                b1 = row * 3 + col
                b2 = row * 3 + col + 1
                inter_pairs.append([b1, b2])
        for row in range(2):
            for col in range(3):
                b1 = row * 3 + col
                b2 = (row + 1) * 3 + col
                inter_pairs.append([b1, b2])
        diagonal_pairs = [
            [0, 4], [1, 4], [2, 4],
            [3, 4], [5, 4],
            [4, 6], [4, 7], [4, 8],
        ]
        inter_pairs.extend(diagonal_pairs)

        print(f"    - Cross-block candidate pairs: {len(inter_pairs)} pairs")
        return torch.tensor(inter_pairs, dtype=torch.long)

    def _precompute_valid_pixels(self):
        valid_pixels = []
        for block_id in range(9):
            pixels = self.block_pixels[block_id]
            valid = pixels[pixels != -1]
            valid_pixels.append(valid)

        print(f"    - ✅ Precomputed valid pixel indices: 9 blocks")
        return valid_pixels

    def _setup_random_sampling_plan(self):
        if self.n_random_pairs == 0:
            return []

        n_block_pairs = self.inter_block_pairs.size(0)
        pairs_per_block = self.n_random_pairs // n_block_pairs
        remainder = self.n_random_pairs % n_block_pairs

        sampling_plan = []
        for bp_idx in range(n_block_pairs):
            n_samples = pairs_per_block + (1 if bp_idx < remainder else 0)
            if n_samples > 0:
                b1, b2 = self.inter_block_pairs[bp_idx].tolist()
                sampling_plan.append((b1, b2, n_samples))

        print(f"    - ✅ Precomputed sampling plan: {len(sampling_plan)} tasks")
        return sampling_plan

    def _sample_inter_block_pairs_optimized(self, batch_size, device):
        """
        ✅ Optimization 3: Batch-parallel random sampling (eliminates nested loops)

        Strategy:
        1. Direct access to precomputed valid pixels (no filtering)
        2. Batch random index generation (vectorized)
        3. Batch indexing operations (GPU-accelerated)

        Parameters:
            batch_size: Batch size
            device: Target device

        Returns: [batch, n_random, 2] - Pixel pair indices
        """
        if self.n_random_pairs == 0:
            return None

        all_pairs = []

        for b1, b2, n_samples in self.sampling_plan:
            pixels_b1 = self.valid_pixels_per_block[b1].to(device)
            pixels_b2 = self.valid_pixels_per_block[b2].to(device)
            n1, n2 = pixels_b1.size(0), pixels_b2.size(0)

            idx1 = torch.randint(0, n1, (batch_size, n_samples), device=device)
            idx2 = torch.randint(0, n2, (batch_size, n_samples), device=device)

            p1 = pixels_b1[idx1]  # [batch, n_samples]
            p2 = pixels_b2[idx2]  # [batch, n_samples]

            pairs = torch.stack([p1, p2], dim=-1)  # [batch, n_samples, 2]
            all_pairs.append(pairs)

        return torch.cat(all_pairs, dim=1)  # [batch, n_random, 2]

    # =================== Mutual Information Analysis Interface ========
    def _get_fixed_features(self, x):
        """
        ✅ New Method: Extract fixed pair quantum outputs separately (for mutual information analysis)

        Purpose:
        - Isolate feature contributions from fixed vs random pairs
        - Calculate I(Fixed; Label) mutual information
        - Validate necessity of fixed pairs

        Input: x [batch, 1, 32, 32]
        Output: [batch, n_fixed*2, num_classes] - Quantum features from fixed pairs only

        Notes:
        - Doesn't affect normal training flow
        - Only called during mutual information analysis
        - Fixed pair indices come from precomputed intra_pair_indices
        """
        batch_size = x.size(0)

        x_flat = x.view(batch_size, -1)  # [batch, 1024]

        fixed_pairs_pixels = x_flat[:, self.intra_pair_indices]  # [batch, n_fixed, 2]

        outputs = []
        for qfuzzier in self.qfuzziers:
            pairs_flat = fixed_pairs_pixels.view(-1, 2)  # [batch*n_fixed, 2]
            qout = qfuzzier(pairs_flat)  # [batch*n_fixed, 2]
            qout = qout.view(batch_size, self.n_fixed_pairs, 2)  # [batch, n_fixed, 2]
            class_output = qout.view(batch_size, -1)  # [batch, n_fixed*2]
            outputs.append(class_output)

        return torch.stack(outputs, dim=-1)  # [batch, n_fixed*2, num_classes]

    def _get_random_features(self, x):
        """
        Purpose:
        - Isolate feature contributions from fixed vs random pairs
        - Calculate I(Random; Label) mutual information
        - Validate incremental information from random pairs

        Input: x [batch, 1, 32, 32]
        Output: [batch, n_random*2, num_classes] - Quantum features from random pairs only

        Notes:
        - Requires dynamic random pair sampling (results vary per call)
        - For fair comparison, recommend consecutive calls on same batch
        - Returns None if n_random_pairs=0
        """
        if self.n_random_pairs == 0:
            return None

        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # [batch, 1024]
        random_pairs_indices = self._sample_inter_block_pairs_optimized(
            batch_size, x.device
        )  # [batch, n_random, 2]
        random_pixels_values = []
        for b in range(batch_size):
            batch_pairs = []
            for pair_idx in range(self.n_random_pairs):
                p1_idx = random_pairs_indices[b, pair_idx, 0].item()
                p2_idx = random_pairs_indices[b, pair_idx, 1].item()
                val1 = x_flat[b, p1_idx]
                val2 = x_flat[b, p2_idx]
                batch_pairs.append([val1.item(), val2.item()])
            random_pixels_values.append(batch_pairs)

        random_pixels_values = torch.tensor(
            random_pixels_values,
            dtype=x.dtype,
            device=x.device
        )  # [batch, n_random, 2]

        outputs = []
        for qfuzzier in self.qfuzziers:
            pairs_flat = random_pixels_values.view(-1, 2)  # [batch*n_random, 2]
            qout = qfuzzier(pairs_flat)  # [batch*n_random, 2]
            qout = qout.view(batch_size, self.n_random_pairs, 2)  # [batch, n_random, 2]
            class_output = qout.view(batch_size, -1)  # [batch, n_random*2]
            outputs.append(class_output)

        return torch.stack(outputs, dim=-1)  # [batch, n_random*2, num_classes]

    def forward(self, x):

        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # [batch, 1024]
        fixed_pairs_pixels = x_flat[:, self.intra_pair_indices]  # [batch, n_fixed, 2]
        if self.n_random_pairs > 0:
            random_pairs_pixels = self._sample_inter_block_pairs_optimized(
                batch_size, x.device
            )  # [batch, n_random, 2]

            all_pairs_pixels = torch.cat([fixed_pairs_pixels, random_pairs_pixels], dim=1)
        else:
            all_pairs_pixels = fixed_pairs_pixels
        # [batch, 512, 2]
        outputs = []
        for qfuzzier in self.qfuzziers:
            pairs_flat = all_pairs_pixels.view(-1, 2)  # [batch*512, 2]
            qout = qfuzzier(pairs_flat)  # [batch*512, 2]
            qout = qout.view(batch_size, self.total_pairs, 2)  # [batch, 512, 2]
            class_output = qout.view(batch_size, -1)  # [batch, 1024]
            outputs.append(class_output)

        return torch.stack(outputs, dim=-1)  # [batch, 1024, 7]


# ==================== Classical CNN Part (Reused from Paper Reproduction Version) ==================
class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class JAFFE_CNN(nn.Module):
    """
    JAFFE Classical CNN Feature Extractor (Paper Description)

    Architecture (Paper pp.497-498):
        Conv1: 1→32, k=5, s=2
        Conv2: 32→64, k=3, s=1
        ResBlock1: 64→64
        Conv3: 64→128, k=3, s=2
        Conv4: 128→128, k=3, s=1
        ResBlock2: 128→128
        Conv5: 128→256, k=3, s=1
        Conv6: 256→512, k=3, s=1
        Flatten → FC
    """

    def __init__(self, output_dim=512):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.res1 = ResidualBlock(64, 64, stride=1)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.res2 = ResidualBlock(128, 128, stride=1)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 32, H/2, W/2]
        x = self.relu(self.bn2(self.conv2(x)))  # [B, 64, H/2, W/2]
        x = self.res1(x)  # [B, 64, H/2, W/2]
        x = self.relu(self.bn3(self.conv3(x)))  # [B, 128, H/4, W/4]
        x = self.relu(self.bn4(self.conv4(x)))  # [B, 128, H/4, W/4]
        x = self.res2(x)  # [B, 128, H/4, W/4]
        x = self.relu(self.bn5(self.conv5(x)))  # [B, 256, H/4, W/4]
        x = self.relu(self.bn6(self.conv6(x)))  # [B, 512, H/4, W/4]

        x = self.avgpool(x)  # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]
        x = self.fc(x)  # [B, output_dim]
        return x


# ==================== JAFFE Complete Network (Modified Version - Dual-Qubit 512-Pair No-Pooling with Batch Parallel Optimization) ==============
class JAFFE_JointMembership(nn.Module):
    """
    JAFFE Dual-Qubit Joint Membership Hybrid Network - No-Pooling Modified Version with Batch Parallel Optimization

    Architecture:
        1. Quantum Branch: Dual-qubit 3x3 grid sampling (512 pairs, no-pooling optimized + batch parallel)
        2. Classical Branch: 6 Conv + 2 ResBlock (paper reproduction)
        3. Fusion Layer: ADD fusion
        4. Classifier: Fully-connected layer

    ✅ Performance Optimizations:
        - Precomputed valid pixel indices (avoids runtime filtering)
        - Batch-parallel random sampling (eliminates nested loops)
        - Vectorized index operations (GPU-accelerated)
        - Precomputed sampling plan (avoids runtime calculations)

    ✅ New Features (2025-01-18):
        - Supports runtime dynamic ratio adjustment
        - Provides mutual information analysis interface

    Modifications (2025-01-19):
        - Pair count: 256 → 512 (matches 1024 pixels, 50% ratio)
        - Grid boundaries: [0,11,22,32] → [0,11,21,32]
        - Fixed pairs: 179 → 358 (~70%)
        - Random pairs: 77 → 154 (~30%, default)
        - Output dimension: [batch, 512, 7] → [batch, 1024, 7]

    Parameters:
        num_classes: Number of classes (JAFFE=7)
        hidden_dim: Hidden layer dimension (recommended 128~512)
        n_qnn_layers: Compatibility parameter (unused)
        n_random_pairs: Random pair count (default=154, ~30%)

    📌 How to Adjust Random Pair Ratio (for mutual information analysis):
        Method 1 - Modify layer default (recommended):
            In __init__ line 55: n_random_pairs=154  # ← Change here
            - 30% random: n_random_pairs=154 (default)
            - 50% random: n_random_pairs=256
            - All random: n_random_pairs=512

        Method 2 - Pass via run.py (overrides default):
            config = Namespace(
                n_random_pairs=256,  # ← Add this line
                ...
            )

        Note: run.py parameters override layer defaults
    """

    def __init__(self, num_classes=7, hidden_dim=512, n_qnn_layers=3, n_random_pairs=154):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.classical_cnn = JAFFE_CNN(output_dim=hidden_dim)
        self.quantum_fuzzy = JointMembership_FuzzyLayer_JAFFE_NoPool(
            num_classes=num_classes,
            n_random_pairs=n_random_pairs
        )

        self.fusion_fc = nn.Linear(num_classes, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, num_classes)
        )
        print(f"\n{'=' * 70}")
        print(
            f"【JAFFE Dual-Qubit Joint Membership Hybrid Network - No-Pooling Modified Version with Batch Parallel Optimization】")
        print(f"{'=' * 70}")
        print(f"  Architecture Components:")
        print(f"    - Classical Branch: 6 Conv + 2 ResBlock (Paper Reproduction)")
        print(f"    - Quantum Branch: Dual-Qubit 3x3 Grid Partition (No-Pooling Modified + Batch Parallel Optimized)")
        print(f"\n  Input Processing:")
        print(f"    - Input Size: 32×32 = 1024 pixels (No Pooling)")
        print(f"    - Direct Processing: Preserves all spatial information")
        print(f"\n  Pair Configuration:")
        print(f"    - Total Pairs: 512 pairs")
        print(f"    - Fixed Pairs: {512 - n_random_pairs} pairs (Intra-block Adjacent)")
        print(f"    - Random Pairs: {n_random_pairs} pairs (Cross-block Dynamic)")
        print(f"\n  Dimension Flow:")
        print(f"    - Quantum Output: [batch, 1024, 7]")
        print(f"    - Fuzzy Rules: [batch, 7]")
        print(f"    - Fusion Dimension: {hidden_dim}")
        print(f"\n  Model Scale:")
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        quantum_params = sum(p.numel() for p in self.quantum_fuzzy.parameters() if p.requires_grad)
        classical_params = sum(p.numel() for p in self.classical_cnn.parameters() if p.requires_grad)
        print(f"    - Total Parameters: {total_params:,}")
        print(f"    - Quantum Branch: {quantum_params:,} ({quantum_params / total_params * 100:.2f}%)")
        print(f"    - Classical Branch: {classical_params:,} ({classical_params / total_params * 100:.2f}%)")
        print(f"\n  ✅ Performance Optimizations:")
        print(f"    - Precomputed valid pixel indices")
        print(f"    - Batch-parallel random sampling")
        print(f"    - Vectorized index operations")
        print(f"    - Precomputed sampling plan")
        print(f"    - Expected speedup: 2-3x")
        print(f"{'=' * 70}\n")

    def get_fused_features(self, x):
        """
        New: Extract fused features (for linear separability analysis)

        Purpose:
        - Calculate Silhouette Score
        - Compute Calinski-Harabasz Score
        - Train Linear Probe

        Input: x [batch, 1, 32, 32]
        Output: [batch, hidden_dim] - Fused feature vectors
        """
        batch_size = x.size(0)
        classical_features = self.classical_cnn(x)
        fuzzy_membership = self.quantum_fuzzy(x)
        fuzzy_membership_clamped = torch.clamp(fuzzy_membership, min=1e-7, max=1.0)
        log_membership = torch.log(fuzzy_membership_clamped)
        fuzzy_features = torch.mean(log_membership, dim=1)
        fuzzy_features = self.fusion_fc(fuzzy_features)

        fused_features = classical_features + fuzzy_features

        return fused_features

    def get_classical_features(self, x):
        return self.classical_cnn(x)

    def get_quantum_features(self, x):
        fuzzy_membership = self.quantum_fuzzy(x)  # [batch, 1024, 7]
        fuzzy_membership_clamped = torch.clamp(fuzzy_membership, min=1e-7, max=1.0)
        log_membership = torch.log(fuzzy_membership_clamped)
        fuzzy_features = torch.mean(log_membership, dim=1)
        fuzzy_features = self.fusion_fc(fuzzy_features)  # [batch, hidden_dim]
        return fuzzy_features

    def forward(self, x):
        batch_size = x.size(0)

        classical_features = self.classical_cnn(x)  # [batch, hidden_dim]

        fuzzy_membership = self.quantum_fuzzy(x)  # [batch, 1024, 7]

        fuzzy_membership_clamped = torch.clamp(fuzzy_membership, min=1e-7, max=1.0)
        log_membership = torch.log(fuzzy_membership_clamped)
        fuzzy_features = torch.mean(log_membership, dim=1)

        fuzzy_features = self.fusion_fc(fuzzy_features)  # [batch, hidden_dim]

        fused_features = classical_features + fuzzy_features  # [batch, hidden_dim]

        output = self.classifier(fused_features)  # [batch, 7]

        return output


# # ==========================FDNN====================================
#
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# def gaussian_membership(x, mean, std):
#     return torch.exp(-((x - mean) ** 2) / (2 * std ** 2))
#
#
# class SingleGaussianMembership(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.mean = None
#         self.std = None
#         self._initialized = False
#
#     def _initialize_params(self, input_dim, device):
#         self.mean = nn.Parameter(torch.randn(input_dim, device=device) * 0.1)
#         self.std = nn.Parameter(torch.abs(torch.randn(input_dim, device=device)) * 0.5 + 0.5)
#         self._initialized = True
#
#     def forward(self, x):
#         if not self._initialized:
#             self._initialize_params(x.shape[1], x.device)
#
#         membership = gaussian_membership(x, self.mean, self.std)
#         return membership
#
#
# class ClassicalFuzzyLayer(nn.Module):
#
#     def __init__(self, num_classes=7, n_layers=3):
#         super().__init__()
#         self.num_classes = num_classes
#
#         self.fuzziers = nn.ModuleList([
#             SingleGaussianMembership() for _ in range(num_classes)
#         ])
#
#     def forward(self, x):
#         outputs = []
#         for fuzzier in self.fuzziers:
#             membership = fuzzier(x)
#             outputs.append(membership)
#
#         stacked = torch.stack(outputs, dim=-1)
#         batch_size, num_features, num_classes = stacked.shape
#         return stacked.view(batch_size * num_features, num_classes)
#
#
# class ResidualBlock(nn.Module):
#
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#
#         self.conv1 = nn.Conv2d(in_channels, out_channels,
#                                kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#
#         self.conv2 = nn.Conv2d(out_channels, out_channels,
#                                kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#         self.downsample = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         identity = self.downsample(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# # ============== 3. JAFFE CNN  ==============
#
# class JAFFE_CNN(nn.Module):
#
#     def __init__(self, output_dim=512):
#         super().__init__()
#
#         # 6个卷积层
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
#         self.bn1 = nn.BatchNorm2d(32)
#
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#
#         self.res1 = ResidualBlock(64, 64, stride=1)
#
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(128)
#
#         self.res2 = ResidualBlock(128, 128, stride=1)
#
#         self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.bn5 = nn.BatchNorm2d(256)
#
#         self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.bn6 = nn.BatchNorm2d(512)
#
#         self.relu = nn.ReLU(inplace=True)
#
#         self.fc1 = None
#         self.fc2 = None
#         self.dropout = nn.Dropout(p=0.7)
#         self.output_dim = output_dim
#
#     def forward(self, x):
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.res1(x)
#         x = self.relu(self.bn3(self.conv3(x)))
#         x = self.relu(self.bn4(self.conv4(x)))
#         x = self.res2(x)
#         x = self.relu(self.bn5(self.conv5(x)))
#         x = self.relu(self.bn6(self.conv6(x)))
#
#         x = x.view(x.size(0), -1)
#
#         if self.fc1 is None:
#             flatten_dim = x.shape[1]
#             self.fc1 = nn.Linear(flatten_dim, 1024).to(x.device)
#             self.fc2 = nn.Linear(1024, self.output_dim).to(x.device)
#
#         x = self.dropout(x)
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#
#         return x
#
#
# # ============== 4. Total FDNN ==============
#
# class JAFFE_JointMembership(nn.Module):
#
#     def __init__(self, num_classes=7, hidden_dim=512, n_qnn_layers=3):
#         super().__init__()
#         self.num_classes = num_classes
#         self.hidden_dim = hidden_dim
#
#         self.classical_cnn = JAFFE_CNN(output_dim=hidden_dim)
#
#         self.classical_fuzzy = ClassicalFuzzyLayer(
#             num_classes=num_classes,
#             n_layers=n_qnn_layers
#         )
#
#         self.fusion_fc = nn.Linear(num_classes, hidden_dim)
#
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.7),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.7),
#             nn.Linear(hidden_dim, num_classes)
#         )
#
#     def forward(self, x):
#         batch_size = x.size(0)
#
#         classical_features = self.classical_cnn(x)  # [batch_size, hidden_dim]
#
#         x_flat = x.view(batch_size, -1)  # [batch_size, C*H*W]
#
#         fuzzy_membership = self.classical_fuzzy(x_flat)  # [batch_size * C*H*W, num_classes]
#
#         fuzzy_membership = fuzzy_membership.view(batch_size, -1, self.num_classes)
#
#         fuzzy_features = torch.prod(fuzzy_membership, dim=1)  # [batch_size, num_classes]
#
#         fuzzy_features = self.fusion_fc(fuzzy_features)  # [batch_size, hidden_dim]
#
#         fused_features = classical_features + fuzzy_features  # [batch_size, hidden_dim]
#
#         output = self.classifier(fused_features)  # [batch_size, num_classes]
#
#         return output
#

# =============================QAHFNN========================================
# """
# JAFFE Single-Qubit Hybrid Neural Network (Paper Reproduction - Dimension Fixed Version)
# Quantum-Assisted Hierarchical Fuzzy Neural Network for JAFFE
#
# Architecture Components:
# 1. Quantum Fuzzy Part: 7 single-qubit QNNs (corresponding to 7 expressions)
# 2. Classical CNN Part: 6 convolutional layers + 2 residual blocks
# 3. Fusion Layer: ADD fusion
# 4. Classifier: Fully-connected layer
# """
#
# import torch
# import torch.nn as nn
# import torchquantum as tq
#
#
# # ============== 1. (Single-QNN-QAHFNN) ==============
# class SingleQubitQNN(tq.QuantumModule):
#     """
#     Single-Qubit Quantum Neural Network (3-layer circuit block)
#
#     Circuit Structure (per layer):
#         R_y(x) → R_z(θ1) → R_y(θ2) → R_z(θ3)
#     Paper Equation (3): μ_A(x_k) = [⟨0|U†(x_k,θ)MU(x_k,θ)|0⟩ + 1] / 2
#     Parameters:
#         n_layers: Number of circuit layers (paper uses 3 layers)
#     """
#
#     def __init__(self, n_layers=3):
#         super().__init__()
#         self.n_wires = 1
#         self.n_layers = n_layers
#
#         self.encoder = tq.GeneralEncoder([
#             {'input_idx': [0], 'func': 'ry', 'wires': [0]}
#         ])
#
#         self.q_layers = nn.ModuleList([
#             self._build_layer() for _ in range(n_layers)
#         ])
#
#         self.measure = tq.MeasureAll(tq.PauliZ)
#
#     def _build_layer(self):
#         return nn.ModuleList([
#             tq.RZ(has_params=True, trainable=True),
#             tq.RY(has_params=True, trainable=True),
#             tq.RZ(has_params=True, trainable=True)
#         ])
#
#     def forward(self, x):
#         bsz = x.shape[0]  # batch*pixels
#
#         device = tq.QuantumDevice(n_wires=1, bsz=bsz, device=x.device)
#
#         for layer in self.q_layers:
#             self.encoder(device, x)
#
#             layer[0](device, wires=0)  # R_z
#             layer[1](device, wires=0)  # R_y
#             layer[2](device, wires=0)  # R_z
#
#         measurement = self.measure(device)  # [batch*pixels, 1]
#
#         qout = (measurement.squeeze() + 1) / 2  # [batch*pixels]
#
#         return qout
#
#     # ============== 2. Quantum Fuzzy Layer (7 membership functions) =====
#
#
# class QuantumFuzzyLayer(nn.Module):
#     """
#     Quantum Fuzzy Layer: k quantum membership functions
#
#     Parameters:
#         num_classes: Number of classes (JAFFE=7)
#         n_layers: Layers per QNN (paper=3)
#     """
#
#     def __init__(self, num_classes=7, n_layers=3):
#         super().__init__()
#         self.num_classes = num_classes
#
#         self.qfuzziers = nn.ModuleList([
#             SingleQubitQNN(n_layers=n_layers) for _ in range(num_classes)
#         ])
#
#     def forward(self, x):
#         batch_size = x.size(0)
#
#         x_flat = x.view(batch_size, -1)  # [batch, H*W]
#         pixel_dim = x_flat.shape[1]  # H*W
#
#         x_all = x_flat.view(-1, 1)  # [batch*H*W, 1]
#
#         outputs = []
#         for qfuzzier in self.qfuzziers:
#             qout = qfuzzier(x_all)  # [batch*H*W]
#
#             qout = qout.view(batch_size, pixel_dim)
#             outputs.append(qout)
#
#         return torch.stack(outputs, dim=-1)
#
#
# # ============== 3. Classical CNN Part (ResNet Variant) ===========
#
# class ResidualBlock(nn.Module):
#
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#
#         self.conv1 = nn.Conv2d(in_channels, out_channels,
#                                kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#
#         self.conv2 = nn.Conv2d(out_channels, out_channels,
#                                kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#         self.downsample = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         identity = self.downsample(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#     # ============== 3. JAFFE CNN (Strictly Following Author's Original Code) ==============
#
#
# class JAFFE_CNN(nn.Module):
#
#     def __init__(self, output_dim=512):
#         super().__init__()
#
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
#         self.bn1 = nn.BatchNorm2d(32)
#
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#
#         self.res1 = ResidualBlock(64, 64, stride=1)
#
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(128)
#
#         self.res2 = ResidualBlock(128, 128, stride=1)
#
#         self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.bn5 = nn.BatchNorm2d(256)
#
#         self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.bn6 = nn.BatchNorm2d(512)
#
#         self.relu = nn.ReLU(inplace=True)
#
#         self.fc1 = None
#         self.fc2 = None
#         self.dropout = nn.Dropout(p=0.5)
#         self.output_dim = output_dim
#
#     def forward(self, x):
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.res1(x)
#         x = self.relu(self.bn3(self.conv3(x)))
#         x = self.relu(self.bn4(self.conv4(x)))
#         x = self.res2(x)
#         x = self.relu(self.bn5(self.conv5(x)))
#         x = self.relu(self.bn6(self.conv6(x)))
#
#         x = x.view(x.size(0), -1)
#
#         if self.fc1 is None:
#             flatten_dim = x.shape[1]
#             self.fc1 = nn.Linear(flatten_dim, 1024).to(x.device)
#             self.fc2 = nn.Linear(1024, self.output_dim).to(x.device)
#
#         x = self.dropout(x)
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#
#         return x
#
#
# # ============== 4. Complete Hybrid Network ==============
# class JAFFE_JointMembership(nn.Module):
#     """
#     JAFFE Single-Qubit Hybrid Network (QA-HFNN) - Dimension Fixed Version
#
#     Architecture:
#         1. Quantum Fuzzy Part: 7 single-qubit QNNs
#         2. Classical Part: 6 Conv + 2 ResBlock
#         3. Fusion Layer: ADD fusion
#         4. Classifier: Fully-connected layer
#
#     Dimension Flow (reference CIFAR-10):
#         Input: [batch, 1, H, W]
#
#         Quantum Branch:
#         → Flatten: [batch, H*W]
#         → Batch: [batch*H*W, 1]
#         → Quantum: [batch*H*W]
#         → Reshape: [batch, H*W]
#         → Stack: [batch, H*W, num_classes]
#         → Rules: [batch, num_classes] ← torch.prod(dim=1)
#         → Fusion: [batch, hidden_dim]
#
#         Classical Branch:
#         → CNN: [batch, hidden_dim]
#
#         Final:
#         → ADD: [batch, hidden_dim]
#         → Classification: [batch, num_classes]
#
#     Parameters:
#         num_classes: Number of classes (JAFFE=7)
#         hidden_dim: Hidden dimension (paper=512)
#         n_qnn_layers: QNN layers (paper=3)
#     """
#
#     def __init__(self, num_classes=7, hidden_dim=512, n_qnn_layers=3):
#         super().__init__()
#         self.num_classes = num_classes
#         self.hidden_dim = hidden_dim
#
#         self.classical_cnn = JAFFE_CNN(output_dim=hidden_dim)
#
#         self.quantum_fuzzy = QuantumFuzzyLayer(
#             num_classes=num_classes,
#             n_layers=n_qnn_layers
#         )
#
#         self.fusion_fc = nn.Linear(num_classes, hidden_dim)
#
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.4),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(hidden_dim, num_classes)
#         )
#
#         print(f"\n{'=' * 70}")
#         print(f"【JAFFE Single-Qubit QA-HFNN Network Configuration】")
#         print(f"{'=' * 70}")
#         print(f"  Dimension Flow:")
#         print(f"    - Input: [batch, 1, H, W]")
#         print(f"    - CNN Features: [batch, {hidden_dim}]")
#         print(f"    - Quantum Flatten: [batch, H*W]")
#         print(f"    - Quantum Batch: [batch*H*W, 1]")
#         print(f"    - Quantum Output: [batch, H*W, {num_classes}]")
#         print(f"    - Fuzzy Rules: [batch, {num_classes}] ← torch.prod(dim=1)")
#         print(f"    - Fusion Layer: [batch, {hidden_dim}]")
#         print(f"    - Final Output: [batch, {num_classes}]")
#         print(f"\n  Model Scale:")
#         total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         quantum_params = sum(p.numel() for p in self.quantum_fuzzy.parameters() if p.requires_grad)
#         classical_params = sum(p.numel() for p in self.classical_cnn.parameters() if p.requires_grad)
#         print(f"    - Total Parameters: {total_params:,}")
#         print(f"    - Quantum Branch: {quantum_params:,} ({quantum_params / total_params * 100:.2f}%)")
#         print(f"    - Classical Branch: {classical_params:,} ({classical_params / total_params * 100:.2f}%)")
#         print(f"{'=' * 70}\n")
#
#     def forward(self, x):
#         batch_size = x.size(0)
#
#         classical_features = self.classical_cnn(x)  # [batch, hidden_dim]
#
#         fuzzy_membership = self.quantum_fuzzy(x)  # [batch, H*W, num_classes]
#
#         fuzzy_features = torch.prod(fuzzy_membership, dim=1)  # [batch, num_classes]
#
#         fuzzy_features = self.fusion_fc(fuzzy_features)  # [batch, hidden_dim]
#
#         fused_features = classical_features + fuzzy_features  # [batch, hidden_dim]
#
#         output = self.classifier(fused_features)  # [batch, num_classes]
#
#         return output
#
#
# # ============= DNN Reproduction =============
# """
# JAFFE Pure DNN Model (Paper Reproduction - Classical CNN Only)
# Maintains same architecture as single-qubit model but removes quantum fuzzy layer
# """
#
# import torch
# import torch.nn as nn
#
#
# # == 1. Residual Block (Same as Single-Qubit Model) ====
# class ResidualBlock(nn.Module):
#
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#
#         self.conv1 = nn.Conv2d(in_channels, out_channels,
#                                kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#
#         self.conv2 = nn.Conv2d(out_channels, out_channels,
#                                kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#         self.downsample = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         identity = self.downsample(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# # ============== 3. JAFFE CNN (Strictly Following Author's Source Code) ==============
# class JAFFE_CNN(nn.Module):
#     def __init__(self, output_dim=512):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
#         self.bn1 = nn.BatchNorm2d(32)
#
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#
#         self.res1 = ResidualBlock(64, 64, stride=1)
#
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(128)
#
#         self.res2 = ResidualBlock(128, 128, stride=1)
#
#         self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.bn5 = nn.BatchNorm2d(256)
#
#         self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.bn6 = nn.BatchNorm2d(512)
#
#         self.relu = nn.ReLU(inplace=True)
#
#         self.fc1 = None
#         self.fc2 = None
#         self.dropout = nn.Dropout(p=0.5)
#         self.output_dim = output_dim
#
#     def forward(self, x):
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.res1(x)
#         x = self.relu(self.bn3(self.conv3(x)))
#         x = self.relu(self.bn4(self.conv4(x)))
#         x = self.res2(x)
#         x = self.relu(self.bn5(self.conv5(x)))
#         x = self.relu(self.bn6(self.conv6(x)))
#
#         x = x.view(x.size(0), -1)
#
#         if self.fc1 is None:
#             flatten_dim = x.shape[1]
#             self.fc1 = nn.Linear(flatten_dim, 1024).to(x.device)
#             self.fc2 = nn.Linear(1024, self.output_dim).to(x.device)
#
#         x = self.dropout(x)
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#
#         return x
#
#
# # === 3. Pure DNN Model (Interface Consistent with Single-Qubit Model) ====
# class JAFFE_JointMembership(nn.Module):
#     """
#     JAFFE Pure DNN Model (Classical CNN Only)
#
#     Architecture:
#         1. Classical Part: 6 Conv + 2 ResBlock
#         2. Classifier: Fully-connected layer
#
#     Parameters:
#         num_classes: Number of classes (JAFFE=7)
#         hidden_dim: Hidden dimension (paper=512)
#         n_qnn_layers: Reserved parameter (interface compatibility, unused)
#     """
#
#     def __init__(self, num_classes=7, hidden_dim=512, n_qnn_layers=3):
#         super().__init__()
#         self.num_classes = num_classes
#         self.hidden_dim = hidden_dim
#
#         self.classical_cnn = JAFFE_CNN(output_dim=hidden_dim)
#
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.4),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(hidden_dim, num_classes)
#         )
#
#     def forward(self, x):
#         classical_features = self.classical_cnn(x)  # [batch_size, hidden_dim]
#
#         output = self.classifier(classical_features)  # [batch_size, num_classes]
#
#         return output
