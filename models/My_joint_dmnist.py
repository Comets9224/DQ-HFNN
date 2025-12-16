
# import torch
# import torch.nn as nn
# import torchquantum as tq
#
#
# # ============ Quantum Circuit: Joint Membership-Single Layer-With Entanglement ===========
# class JointMembership_QuantumCircuit(nn.Module):
#     """
#     Joint Membership Quantum Circuit Model (Reusable Version)
#     - 2 qubits
#     - 1 layer (single data encoding)
#     - Contains entanglement gate (CNOT)
#     - **Simultaneous measurement of both qubits**, outputs 2D joint membership
#     """
#
#     class QLayer(tq.QuantumModule):
#         def __init__(self):
#             super().__init__()
#             self.n_wires = 2
#
#             self.rz0 = tq.RZ(has_params=True, trainable=True)
#             self.ry0 = tq.RY(has_params=True, trainable=True)
#
#             self.cnot1 = tq.CNOT()
#             self.cnot2 = tq.CNOT()
#
#         def forward(self, device: tq.QuantumDevice):
#             self.rz0(device, wires=0)
#             self.cnot1(device, wires=[0, 1])
#             self.ry0(device, wires=0)
#             self.cnot2(device, wires=[0, 1])
#
#     def __init__(self):
#         super().__init__()
#         self.n_wires = 2
#         self.encoder = tq.GeneralEncoder([
#             {'input_idx': [0], 'func': 'ry', 'wires': [0]},
#             {'input_idx': [1], 'func': 'ry', 'wires': [1]},
#         ])
#         self.q_layer = self.QLayer()
#         self.measure = tq.MeasureAll(tq.PauliZ)
#
#     def forward(self, x, use_qiskit=False):
#         bsz = x.shape[0]
#         assert x.shape[1] == 2, f"Expected input dimension 2, got {x.shape[1]}"
#         device = tq.QuantumDevice(n_wires=2, bsz=bsz, device=x.device)
#
#         self.encoder(device, x)
#         self.q_layer(device)
#
#         measurements = self.measure(device)  # shape: [batch, 2]
#         qout = (measurements + 1) / 2
#         return qout  # [batch, 2]
#
#
# # ==================== Quantum Fuzzy Layer: 3x3 Grid Partition Sampling (No-Pooling Modified Version - Batch Parallelism Optimized) ==========
# class JointMembership_FuzzyLayer(nn.Module):
#     """
#     Joint Membership Quantum Fuzzy Layer with 3x3 Grid Partition Sampling for DirtyMNIST (No-Pooling Modified Version - Batch Parallelism Optimized)
#
#     ✅ Key Improvements (vs Pooling Version):
#     1. **No Pooling**: Directly processes 28×28 = 784 pixels
#     2. **Grid Boundary Adjustment**: [0, 9, 18, 28] (adapted for 28×28)
#     3. **Pair Count Expansion**: 98 pairs → 392 pairs (4x increase, matches pixel growth)
#     4. **Batch Parallelism Optimization**: Precomputed indices + vectorized sampling
#
#     Grid Partition Details (28×28):
#     - Block Boundaries: [0, 9, 18, 28]
#     - Block Size Distribution:
#       * 4 corner blocks: 9×9 = 81 pixels
#       * 4 edge blocks: 9×10 = 90 pixels
#       * 1 center block: 10×10 = 100 pixels
#     - Total Pixels: 4×81 + 4×90 + 100 = 784✓
#
#     Modified Configuration:
#     - Total Pairs: 392 pairs (~50% pixel coverage)
#     - Fixed Pairs: 275 pairs (~70%, intra-block 2D adjacency)
#     - Random Pairs: 117 pairs (~30%, cross-block long-range, default)
#     - Output Dimension: [batch, 392*2, num_classes] = [batch, 784, 10]
#
#     Parameters:
#     - num_classes: Number of classes (DirtyMNIST=10)
#     - n_random_pairs: Random pair count (default=117, ~30%)
#     """
#
#     def __init__(self, num_classes=10, n_random_pairs=117):
#         super(JointMembership_FuzzyLayer, self).__init__()
#         self.num_classes = num_classes
#
#         self.pooled_size = 28
#         self.pixel_dim = 784
#
#         self.grid_size = 3
#         self.block_sizes = [0, 9, 18, 28]
#
#         self.total_pairs = 117
#         self.n_random_pairs = n_random_pairs
#         self.n_fixed_pairs = self.total_pairs - n_random_pairs
#
#         assert 0 <= n_random_pairs <= self.total_pairs, f"n_random_pairs={n_random_pairs}超出总配对数{self.total_pairs}"
#
#         self.register_buffer('block_pixels', self._compute_block_partition())
#
#         if self.n_fixed_pairs > 0:
#             all_intra_pairs = self._compute_all_intra_pairs()
#
#             self.register_buffer('intra_pair_indices',
#                                  self._sample_fixed_pairs(all_intra_pairs, self.n_fixed_pairs))
#         else:
#
#             self.register_buffer('intra_pair_indices', torch.empty(0, 2, dtype=torch.long))
#
#         self.register_buffer('inter_block_pairs', self._compute_inter_block_pairs())
#
#         self.valid_pixels_per_block = self._precompute_valid_pixels()
#
#         self.sampling_plan = self._setup_random_sampling_plan()
#
#         self.qfuzziers = nn.ModuleList([
#             JointMembership_QuantumCircuit() for _ in range(num_classes)
#         ])
#         print(f"\n{'=' * 70}")
#         print(
#             f"【DirtyMNIST 3x3 Grid Quantum Fuzzy Layer (No-Pooling Modified - Dual-Qubit - Batch Parallel Optimized)】")
#         print(f"{'=' * 70}")
#         print(f"  Pooling Strategy: No pooling (direct raw image processing)")
#         print(f"  Input Size: 28×28 = 784 pixels")
#         print(f"\n  Grid Configuration:")
#         print(f"    - Partitions: 3×3 = 9 blocks")
#         print(f"    - Block Boundaries: {self.block_sizes} (adapted for 28×28)")
#         self._print_block_info()
#         print(f"\n  Pairing Strategy (Total Pairs={self.total_pairs}):")
#         print(f"    - Fixed Pairs: {self.n_fixed_pairs} pairs (intra-block 2D spatial adjacency)")
#         print(f"    - Random Pairs: {self.n_random_pairs} pairs (cross-block dynamic sampling)")
#         print(f"    - Total Pairs: {self.total_pairs} pairs")
#         print(
#             f"    - Pair Ratio: {self.n_fixed_pairs / self.total_pairs * 100:.1f}% fixed + {self.n_random_pairs / self.total_pairs * 100:.1f}% random")
#         print(f"\n  Quantum Computation:")
#         print(f"    - Quantum Circuits: {num_classes} (independent per class)")
#         print(f"    - Circuit Calls/Sample: {self.total_pairs * num_classes}")
#         print(f"    - Output Dimension: [batch, {self.total_pairs * 2}, {num_classes}]")
#         print(f"\n  ✅ Performance Optimizations:")
#         print(f"    - Precomputed valid pixel indices")
#         print(f"    - Batch-parallel random sampling")
#         print(f"    - Vectorized index operations")
#         print(f"    - Precomputed sampling plan")
#         print(f"{'=' * 70}\n")
#
#     def _print_block_info(self):
#         """Print 3x3 grid block details"""
#         print(f"    - Block Size Distribution:")
#         block_names = [
#             "Top-left", "Top", "Top-right",
#             "Left", "Center", "Right",
#             "Bottom-left", "Bottom", "Bottom-right"
#         ]
#         total_pixels = 0
#         for i in range(9):
#             row_idx = i // 3
#             col_idx = i % 3
#             h = self.block_sizes[row_idx + 1] - self.block_sizes[row_idx]
#             w = self.block_sizes[col_idx + 1] - self.block_sizes[col_idx]
#             pixels = h * w
#             total_pixels += pixels
#             print(f"      * Block {i} ({block_names[i]:4s}): {h}×{w} = {pixels:3d} pixels")
#         print(f"    - Total Pixel Verification: {total_pixels} (should be 784)")
#
#     def _compute_block_partition(self):
#
#         block_pixels_list = []
#         max_size = 0
#
#         for block_row in range(self.grid_size):
#             for block_col in range(self.grid_size):
#                 pixels = []
#                 row_start = self.block_sizes[block_row]
#                 row_end = self.block_sizes[block_row + 1]
#                 col_start = self.block_sizes[block_col]
#                 col_end = self.block_sizes[block_col + 1]
#
#                 for r in range(row_start, row_end):
#                     for c in range(col_start, col_end):
#                         pixel_idx = r * self.pooled_size + c
#                         pixels.append(pixel_idx)
#
#                 block_pixels_list.append(pixels)
#                 max_size = max(max_size, len(pixels))
#
#         block_pixels_padded = []
#         for pixels in block_pixels_list:
#             padded = pixels + [-1] * (max_size - len(pixels))
#             block_pixels_padded.append(padded)
#
#         return torch.tensor(block_pixels_padded, dtype=torch.long)
#
#     def _compute_all_intra_pairs(self):
#
#         all_pairs = []
#
#         for block_id in range(9):
#             block_row = block_id // 3
#             block_col = block_id % 3
#
#             row_start = self.block_sizes[block_row]
#             row_end = self.block_sizes[block_row + 1]
#             col_start = self.block_sizes[block_col]
#             col_end = self.block_sizes[block_col + 1]
#             for r in range(row_start, row_end):
#                 for c in range(col_start, col_end - 1):
#                     idx1 = r * self.pooled_size + c
#                     idx2 = r * self.pooled_size + c + 1
#                     all_pairs.append((idx1, idx2))
#             for r in range(row_start, row_end - 1):
#                 for c in range(col_start, col_end):
#                     idx1 = r * self.pooled_size + c
#                     idx2 = (r + 1) * self.pooled_size + c
#                     all_pairs.append((idx1, idx2))
#
#         print(f"    - Intra-block adjacent pair pool: {len(all_pairs)} pairs (horizontal+vertical)")
#         return all_pairs
#
#     def _sample_fixed_pairs(self, all_pairs, n_fixed):
#         """
#         Uniformly sample fixed pairs from all possible pairs to ensure full image coverage
#
#         Strategy: Use fixed random seed to ensure training reproducibility
#         Parameters:
#             all_pairs: List[Tuple] - All candidate pairs
#             n_fixed: int - Number of fixed pairs to sample
#
#         Returns: [n_fixed, 2] tensor
#         """
#         # Use fixed seed to ensure reproducibility
#         rng = torch.Generator().manual_seed(42)
#         indices = torch.randperm(len(all_pairs), generator=rng)[:n_fixed]
#
#         sampled_pairs = [all_pairs[i] for i in indices]
#
#         covered_blocks = set()
#         for p1, p2 in sampled_pairs:
#             for pixel in [p1, p2]:
#                 row = pixel // self.pooled_size
#                 col = pixel % self.pooled_size
#                 # Determine block position based on new boundaries
#                 block_row = 0 if row < 9 else (1 if row < 18 else 2)
#                 block_col = 0 if col < 9 else (1 if col < 18 else 2)
#                 covered_blocks.add(block_row * 3 + block_col)
#
#         print(f"    - Fixed pairs cover blocks: {len(covered_blocks)}/9")
#
#         return torch.tensor(sampled_pairs, dtype=torch.long)
#
#     def _compute_inter_block_pairs(self):
#         """
#         Precompute candidate block pairs for cross-block sampling (adjacent+diagonal)
#
#         Returns: [n_block_pairs, 2] - Block pair indices
#         """
#         inter_pairs = []
#
#         for row in range(3):
#             for col in range(2):
#                 b1 = row * 3 + col
#                 b2 = row * 3 + col + 1
#                 inter_pairs.append([b1, b2])
#
#         for row in range(2):
#             for col in range(3):
#                 b1 = row * 3 + col
#                 b2 = (row + 1) * 3 + col
#                 inter_pairs.append([b1, b2])
#
#         diagonal_pairs = [
#             [0, 4], [1, 4], [2, 4],
#             [3, 4], [5, 4],
#             [4, 6], [4, 7], [4, 8],
#         ]
#         inter_pairs.extend(diagonal_pairs)
#
#         print(f"    - Cross-block candidate pairs: {len(inter_pairs)} pairs")
#         return torch.tensor(inter_pairs, dtype=torch.long)
#
#     def _precompute_valid_pixels(self):
#         """
#         ✅ Optimization 1: Precompute valid pixel indices for each block (excluding padding -1)
#         Avoids runtime repeated filtering operations
#
#         Returns: List[Tensor] - Valid pixels for 9 blocks (varying lengths, cannot stack)
#         """
#         valid_pixels = []
#         for block_id in range(9):
#             pixels = self.block_pixels[block_id]
#             valid = pixels[pixels != -1]
#             valid_pixels.append(valid)
#
#         print(f"    - ✅ Precomputed valid pixel indices: 9 blocks")
#         return valid_pixels
#
#     def _setup_random_sampling_plan(self):
#         """
#         ✅ Optimization 2: Precompute random sampling plan (avoids runtime allocation)
#
#         Returns: List[Tuple[int, int, int]] - (block1, block2, n_samples)
#         """
#         if self.n_random_pairs == 0:
#             return []
#
#         n_block_pairs = self.inter_block_pairs.size(0)
#         pairs_per_block = self.n_random_pairs // n_block_pairs
#         remainder = self.n_random_pairs % n_block_pairs
#
#         sampling_plan = []
#         for bp_idx in range(n_block_pairs):
#             n_samples = pairs_per_block + (1 if bp_idx < remainder else 0)
#             if n_samples > 0:
#                 b1, b2 = self.inter_block_pairs[bp_idx].tolist()
#                 sampling_plan.append((b1, b2, n_samples))
#
#         print(f"    - ✅ Precomputed sampling plan: {len(sampling_plan)} tasks")
#         return sampling_plan
#
#     def _sample_inter_block_pairs_optimized(self, batch_size, device):
#         """
#         ✅ Optimization 3: Batch-parallel random sampling (eliminates nested loops)
#
#         Strategy:
#         1. Direct access to precomputed valid pixels (no filtering)
#         2. Batch random index generation (vectorized)
#         3. Batch indexing operations (GPU-accelerated)
#
#         Parameters:
#             batch_size: Batch size
#             device: Target device
#
#         Returns: [batch, n_random, 2] - Pixel pair indices
#         """
#         if self.n_random_pairs == 0:
#             return None
#
#         all_pairs = []
#
#         for b1, b2, n_samples in self.sampling_plan:
#             pixels_b1 = self.valid_pixels_per_block[b1].to(device)
#             pixels_b2 = self.valid_pixels_per_block[b2].to(device)
#             n1, n2 = pixels_b1.size(0), pixels_b2.size(0)
#
#             idx1 = torch.randint(0, n1, (batch_size, n_samples), device=device)
#             idx2 = torch.randint(0, n2, (batch_size, n_samples), device=device)
#
#             p1 = pixels_b1[idx1]  # [batch, n_samples]
#             p2 = pixels_b2[idx2]  # [batch, n_samples]
#
#             pairs = torch.stack([p1, p2], dim=-1)  # [batch, n_samples, 2]
#             all_pairs.append(pairs)
#
#         return torch.cat(all_pairs, dim=1)  # [batch, n_random, 2]
#
#     def forward(self, x):
#
#         batch_size = x.size(0)
#
#         x_flat = x.view(batch_size, -1)  # [batch, 784]
#
#         if self.n_fixed_pairs > 0:
#             fixed_pairs_pixels = x_flat[:, self.intra_pair_indices]  # [batch, n_fixed, 2]
#         else:
#             fixed_pairs_pixels = None
#
#         if self.n_random_pairs > 0:
#             random_pairs_pixels = self._sample_inter_block_pairs_optimized(
#                 batch_size, x.device
#             )  # [batch, n_random, 2]
#
#             if fixed_pairs_pixels is not None:
#                 all_pairs_pixels = torch.cat([fixed_pairs_pixels, random_pairs_pixels], dim=1)
#             else:
#                 all_pairs_pixels = random_pairs_pixels
#         else:
#             all_pairs_pixels = fixed_pairs_pixels
#         # [batch, total_pairs, 2]
#
#         outputs = []
#         for qfuzzier in self.qfuzziers:
#             pairs_flat = all_pairs_pixels.view(-1, 2)  # [batch*total_pairs, 2]
#             qout = qfuzzier(pairs_flat)  # [batch*total_pairs, 2]
#             qout = qout.view(batch_size, self.total_pairs, 2)  # [batch, total_pairs, 2]
#             class_output = qout.view(batch_size, -1)  # [batch, total_pairs*2]
#             outputs.append(class_output)
#
#         return torch.stack(outputs, dim=-1)  # [batch, total_pairs*2, num_classes]
#
#
# # ==================== Classical CNN Feature Extractor ====================
# class classical_layer_dmnist(nn.Module):
#     """
#     Classical CNN Feature Extractor for Dirty MNIST (maintains original interface)
#     Input: [batch, 1, 28, 28] (grayscale images)
#     Output: [batch, fusion_dim] (configurable feature dimension)
#     """
#
#     def __init__(self, num_classes=10, fusion_dim=128):
#         super(classical_layer_dmnist, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=10,
#                                kernel_size=5, stride=1, padding=0)
#         self.maxpool1 = nn.MaxPool2d(2)
#
#         self.conv2 = nn.Conv2d(in_channels=10, out_channels=20,
#                                kernel_size=5, stride=1, padding=0)
#         self.maxpool2 = nn.MaxPool2d(2)
#
#         self.linear1 = nn.Linear(320, fusion_dim)
#
#         self.relu = nn.LeakyReLU()
#         self.dropout = nn.Dropout(p=0.3)
#
#     def forward(self, x):
#         x = self.relu(self.maxpool1(self.conv1(x)))  # [batch, 10, 12, 12]
#         x = self.relu(self.maxpool2(self.conv2(x)))  # [batch, 20, 4, 4]
#         x = x.view(x.size(0), -1)  # [batch, 320]
#         x = self.relu(self.linear1(x))  # [batch, fusion_dim]
#         x = self.dropout(x)
#         return x
#
#
# # ==================== Dirty MNIST Complete Network (No-Pooling Modified Version - Batch Parallel Optimized) ===================
# class DirtyMnist_JointMembership(nn.Module):
#     """
#     Dirty MNIST Dual-Qubit Joint Membership Hybrid Network - No-Pooling Modified Version with Batch Parallel Optimization
#
#     Architecture:
#         1. Quantum Branch: Dual-qubit 3x3 grid sampling (configurable fixed/random ratio, no-pooling optimization + batch parallelism)
#         2. Classical Branch: Standard CNN (maintains original architecture)
#         3. Fusion Layer: ADD fusion
#         4. Classifier: Fully-connected layer
#
#     ✅ Performance Optimizations (referencing JAFFE):
#         - Precomputed valid pixel indices (avoids runtime filtering)
#         - Batch-parallel random sampling (eliminates nested loops)
#         - Vectorized index operations (GPU-accelerated)
#         - Precomputed sampling plan (avoids runtime calculations)
#
#     Modifications:
#         - Removed pooling: 28×28→28×28 (preserves all pixels)
#         - Pair count: 98 pairs → 392 pairs (4x expansion)
#         - Grid boundaries: [0,5,10,14] → [0,9,18,28]
#         - Fixed/random pair ratio adjustable via n_random_pairs parameter
#         - Output dimension: [batch, 196, 10] → [batch, 784, 10]
#
#     Parameters:
#         num_classes: Number of classes (DirtyMNIST=10)
#         fusion_dim: Fusion layer dimension (recommended 128)
#         n_random_pairs: Random pair count (default=117, ~30%; set to 392 for fully random)
#
#     📌 Interface Compatibility:
#         - Maintains class name DirtyMnist_JointMembership (same as original)
#         - Preserves parameter names and defaults
#         - Plug-and-play, no run script modifications needed
#     """
#
#     def __init__(self, num_classes=10, fusion_dim=128, n_random_pairs=117):
#         super(DirtyMnist_JointMembership, self).__init__()
#         self.k = num_classes
#
#         self.class_layer = classical_layer_dmnist(num_classes, fusion_dim)
#
#         self.qfuzzy_layer = JointMembership_FuzzyLayer(
#             num_classes=num_classes,
#             n_random_pairs=n_random_pairs
#         )
#
#         self.flinear = nn.Linear(self.k, fusion_dim)
#
#         self.classi = nn.Linear(fusion_dim, self.k)
#
#         print(f"\n{'=' * 70}")
#         print(
#             f"【DirtyMNIST Dual-Qubit Joint Membership Hybrid Network - No-Pooling Modified Version with Batch Parallel Optimization】")
#         print(f"{'=' * 70}")
#         print(f"  Architecture Components:")
#         print(f"    - Classical Branch: Standard CNN Feature Extractor")
#         print(f"    - Quantum Branch: Dual-Qubit 3x3 Grid Partition (No-Pooling Modified + Batch Parallel Optimized)")
#         print(f"\n  Input Processing:")
#         print(f"    - Input Size: 28×28 = 784 pixels (no pooling)")
#         print(f"    - Direct Processing: Preserves all spatial information")
#         print(f"\n  Pair Configuration:")
#         print(f"    - Total Pairs: 392 pairs")
#         print(f"    - Fixed Pairs: {392 - n_random_pairs} pairs (intra-block adjacency)")
#         print(f"    - Random Pairs: {n_random_pairs} pairs (cross-block dynamic)")
#         print(f"\n  Dimension Flow:")
#         print(f"    - Quantum Output: [batch, 784, 10]")
#         print(f"    - Fuzzy Rules: [batch, 10]")
#         print(f"    - Fusion Dimension: {fusion_dim}")
#         print(f"\n  Model Scale:")
#         total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         quantum_params = sum(p.numel() for p in self.qfuzzy_layer.parameters() if p.requires_grad)
#         classical_params = sum(p.numel() for p in self.class_layer.parameters() if p.requires_grad)
#         print(f"    - Total Parameters: {total_params:,}")
#         print(f"    - Quantum Branch: {quantum_params:,} ({quantum_params / total_params * 100:.2f}%)")
#         print(f"    - Classical Branch: {classical_params:,} ({classical_params / total_params * 100:.2f}%)")
#         print(f"\n  ✅ Performance Optimizations:")
#         print(f"    - Precomputed valid pixel indices")
#         print(f"    - Batch-parallel random sampling")
#         print(f"    - Vectorized index operations")
#         print(f"    - Precomputed sampling plan")
#         print(f"    - Expected speedup: 2-3x")
#         print(f"\n  ✅ Interface Compatibility:")
#         print(f"    - Maintained class name: DirtyMnist_JointMembership")
#         print(f"    - Preserved parameter names: num_classes, fusion_dim, n_random_pairs")
#         print(f"    - Plug-and-play, no run script modifications needed")
#         print(f"{'=' * 70}\n")
#
#     def forward(self, x):
#         batch_size = x.size(0)
#
#         c_part = self.class_layer(x)  # [batch, fusion_dim]
#
#         fuzzy_output = self.qfuzzy_layer(x)  # [batch, total_pairs*2, 10]
#
            # fuzzy_membership_clamped = torch.clamp(fuzzy_membership, min=1e-7, max=1.0)
            # log_membership = torch.log(fuzzy_membership_clamped)
            # fuzzy_features = torch.mean(log_membership, dim=1)
#
#         fusion_output = torch.add(c_part, self.flinear(fuzzy_rule_output))  # [batch, fusion_dim]
#
#         output = self.classi(fusion_output)  # [batch, 10]
#
#         return output
#
#
# # == == == == == == == == == == == == == =FDNN == == == == == == == == == == == == == == == == ==
# """
# Dirty MNIST-specific FDNN Architecture (Classical Gaussian Fuzzy Function)
# - Objective: Classical benchmark for quantum method comparison
# - Architecture: Gaussian membership function + classical fuzzy logic
# - Processing: Pixel-by-pixel processing of 784 pixels (no pooling)
# - Pixel-level independent parameters: Each pixel has independent mean and std per class
# - Class name: Fully compatible with main program naming conventions (drop-in replacement for QAHFNN)
# """
#
# import torch
# import torch.nn as nn
# import torch.nn.init as init
#
#
# # ==================== 经典模糊层（替换单量子比特电路 - 像素级独立参数）====================
# class JointMembership_FuzzyLayer(nn.Module):
#     """
#     Classical Fuzzy Layer for Dirty MNIST (replaces quantum layer)
#
#     Key Features:
#     1. **No pooling**: Directly processes 28×28=784 pixels
#     2. **Pixel-wise processing**: Each pixel independently passes through Gaussian membership function
#     3. **Pixel-level independent parameters**: Independent mean and std per pixel per class
#     4. **Batch parallelism**: All pixels processed in one batch
#
#     Input: [batch, 1, 28, 28] - Raw images
#     Output: [batch, 784, num_classes] - Membership degree per pixel per class
#
#     Parameter count: 784 pixels × num_classes × 2 parameters (mean+std) = 15,680 parameters (for 10 classes)
#     """
#
#     def __init__(self, num_classes):
#         super(JointMembership_FuzzyLayer, self).__init__()
#         self.num_classes = num_classes
#         self.pixel_dim = 784  # 28×28
#
#         self.means = nn.Parameter(torch.randn(self.pixel_dim, num_classes) * 0.5)
#         self.stds = nn.Parameter(torch.ones(self.pixel_dim, num_classes) * 0.5)
#
#         print(f"\n【Classical Architecture: Gaussian Fuzzy Layer - Pixel-Level Independent Parameters】")
#         print(f"  - Input dimension: 28×28 = {self.pixel_dim} pixels")
#         print(f"  - Pooling strategy: No pooling (direct processing)")
#         print(f"  - Parameter granularity: Pixel-level independent parameters (matches paper design)")
#         print(f"  - Per-pixel per-class: Independent mean + std")
#         print(f"\n  Parameter Scale:")
#         print(f"    - Means parameters: [{self.pixel_dim}, {num_classes}]")
#         print(f"    - Stds parameters: [{self.pixel_dim}, {num_classes}]")
#         print(f"    - Total parameters: {self.pixel_dim} × {num_classes} × 2 = {self.pixel_dim * num_classes * 2:,}")
#         print(f"\n  Computation Flow:")
#         print(f"    - Input flattening: [batch, 1, 28, 28] → [batch, 784]")
#         print(f"    - Gaussian calculation: Independent membership per pixel per class")
#         print(f"    - Output dimension: [batch, {self.pixel_dim}, {num_classes}]")
#         print(f"    - GPU parallelism: 100% (batch vectorization)")
#         print(f"\n  Comparison:")
#         print(f"    - Class-level parameters (incorrect): {num_classes} × 2 = {num_classes * 2}")
#         print(f"    - Pixel-level parameters (correct): {self.pixel_dim * num_classes * 2:,}")
#         print(f"    - Parameter multiple: {(self.pixel_dim * num_classes * 2) / (num_classes * 2):.0f}x\n")
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         x_flat = x.view(batch_size, -1)  # [batch, 784]
#         # [batch, 784] → [batch, 784, 1]
#         x_expanded = x_flat.unsqueeze(-1)  # [batch, 784, 1]
#
#         # x_expanded: [batch, 784, 1]
#         # self.means: [784, num_classes]
#         # self.stds: [784, num_classes]
#         # broader: [batch, 784, 1] - [784, num_classes] → [batch, 784, num_classes]
#
#         membership = torch.exp(
#             -((x_expanded - self.means) ** 2) / (2 * self.stds ** 2 + 1e-8)
#         )  # [batch, 784, num_classes]
#
#         return membership
#
#
# # ==================== Classical CNN Feature Extractor (Identical to QAHFNN) ====================
# class classical_layer_dmnist(nn.Module):
#     """
#     Classical CNN Feature Extractor for Dirty MNIST
#     Input: [batch, 1, 28, 28] (grayscale images)
#     Output: [batch, fusion_dim] (configurable feature dimension)
#
#     Architecture:
#     - Conv1: 28×28×1 → 12×12×10 (kernel=5, maxpool=2)
#     - Conv2: 12×12×10 → 4×4×20 (kernel=5, maxpool=2)
#     - FC: 320 → fusion_dim
#     """
#
#     def __init__(self, num_classes=10, fusion_dim=128):
#         super(classical_layer_dmnist, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=10,
#                                kernel_size=5, stride=1, padding=0)
#         self.maxpool1 = nn.MaxPool2d(2)
#         self.conv2 = nn.Conv2d(in_channels=10, out_channels=20,
#                                kernel_size=5, stride=1, padding=0)
#         self.maxpool2 = nn.MaxPool2d(2)
#         self.linear1 = nn.Linear(320, fusion_dim)
#         self.relu = nn.LeakyReLU()
#         self.dropout = nn.Dropout(p=0.3)
#
#     def forward(self, x):
#         x = self.relu(self.maxpool1(self.conv1(x)))  # [batch, 10, 12, 12]
#         x = self.relu(self.maxpool2(self.conv2(x)))  # [batch, 20, 4, 4]
#         x = x.view(x.size(0), -1)  # [batch, 320]
#         x = self.relu(self.linear1(x))  # [batch, fusion_dim]
#         x = self.dropout(x)
#
#         return x
#
#
# # ================ Complete FDNN Network (Fully Compatible with QAHFNN Interface - Pixel-Level Independent Parameters) ===========
# class DirtyMnist_JointMembership(nn.Module):
#     """
#     Classical Fuzzy Deep Neural Network (FDNN) for Dirty MNIST (Gaussian Membership Function - Pixel-Level Independent Parameters)
#
#     Architecture Components:
#     1. Classical Branch: CNN Feature Extractor (specialized for grayscale images)
#     2. Classical Fuzzy Branch: Gaussian Membership Function (pixel-level independent parameters, no pooling, processes 784 pixels individually)
#     3. Fusion Layer: Classical features + Fuzzy features
#     4. Classifier: Fully-connected layer
#
#     Fuzzy Configuration (Classical Method - Pixel-Level Independent Parameters):
#     - **Gaussian Function**: Independent membership function per pixel per class
#     - **Trainable Parameters**: Independent mean + std per pixel per class
#     - **Total Parameters**: 784 pixels × 10 classes × 2 parameters = 15,680 fuzzy parameters
#     - **No Pooling**: Direct processing of 784 pixels
#     - **Pixel-wise Processing**: Independent membership calculation per pixel
#
#     Key Dimension Transformations:
#     - Input: [batch, 1, 28, 28]
#     - Flatten: [batch, 784]
#     - Fuzzy Output: [batch, 784, 10]
#     - Fuzzy Rules: [batch, 10]
#     - Fusion: [batch, 128]
#     - Classification: [batch, 10]
#
#     Computational Cost:
#     - Gaussian Function Calculations/Sample: 784 × 10 = 7,840
#     - Estimated Training Time: ~5-8 minutes/epoch (faster than quantum method)
#
#     Fuzzy Logic:
#     - Implements fuzzy "AND" using torch.prod
#     - Maintains consistency with original QAHFNN implementation
#     - ⚠️ Warning: Multiplying 784 values may cause numerical underflow and vanishing gradients
#
#     Parameters:
#     - num_classes: Number of classes (default 10)
#     - fusion_dim: Fusion layer dimension (recommended 128~256)
#     - n_random_pairs: Compatibility parameter (ignored in FDNN)
#     """
#
#     def __init__(self, num_classes=10, fusion_dim=128, n_random_pairs=0):
#         super(DirtyMnist_JointMembership, self).__init__()
#         self.k = num_classes
#
#         # ⚠️ n_random_pairs parameter is ignored (FDNN architecture doesn't use pairing)
#         if n_random_pairs != 0:
#             print(f"  ⚠️  警告: n_random_pairs={n_random_pairs} 在FDNN架构中被忽略")
#
#         # Classical CNN feature extraction (specialized for grayscale images)
#         self.class_layer = classical_layer_dmnist(num_classes, fusion_dim)
#
#         # Classical Gaussian Fuzzy Layer (replaces quantum layer - pixel-level independent parameters)
#         self.qfuzzy_layer = JointMembership_FuzzyLayer(num_classes=num_classes)
#
#         # Fusion and Classification Layer
#         self.flinear = nn.Linear(self.k, fusion_dim)
#         self.classi = nn.Linear(fusion_dim, self.k)
#
#         print(
#             f"\n【Dirty MNIST FDNN Network Configuration (Classical Gaussian Fuzzy - Pixel-Level Independent Parameters)】")
#         print(f"  - Classical branch: CNN → {fusion_dim}D")
#         print(f"  - Fuzzy branch: Pixel-level Gaussian×784 → {num_classes}D")
#         print(f"  - Membership function: Gaussian (pixel-level independent mean + std)")
#         print(f"  - Fuzzy rules: torch.prod (original AND logic)")
#         print(f"  - Fusion method: torch.add")
#         print(f"  - Classifier: {fusion_dim} → {num_classes}")
#         print(f"\n  Fuzzy layer parameters:")
#         print(f"    - Pixel-level means: [784, {num_classes}] = {784 * num_classes:,}")
#         print(f"    - Pixel-level stds: [784, {num_classes}] = {784 * num_classes:,}")
#         print(f"    - Total fuzzy parameters: {784 * num_classes * 2:,} (matches paper design)")
#         print(f"\n  Computational cost:")
#         print(f"    - Gaussian calculations/sample: {784 * num_classes}")
#         print(f"    - Expected accuracy: ~83-85% (classical benchmark)")
#         print(f"    - Comparison method: QAHFNN (quantum approach)\n")
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         c_part = self.class_layer(x)  # [batch, fusion_dim]
#         fuzzy_output = self.qfuzzy_layer(x)  # [batch, 784, 10]
#         fuzzy_rule_output = torch.prod(fuzzy_output, dim=1)  # [batch, 10]
#
#         fusion_output = torch.add(c_part, self.flinear(fuzzy_rule_output))
#
#         output = self.classi(fusion_output)
#         return output
#

# ==================================QAHFNN===========================
# """
# Dirty MNIST-specific single-qubit architecture (full reproduction of the author's paper)
# - Goal: Achieve the paper's benchmark of 85% validation accuracy
# - Architecture: Single-qubit + 3-layer data-reuploading
# - Processing: Pixel-by-pixel processing of 784 pixels (no pooling)
# - Class name: Fully compatible with the main program's naming conventions
# """
#
# import torch
# import torch.nn as nn
# import torchquantum as tq
#
#
# # ==================== Author's single-qubit circuit (3-layer data-reuploading) =====================
# class SingleQubit_QuantumCircuit(nn.Module):
#     """
#     Single-qubit circuit from the paper (3-layer data-reuploading)
#
#     Circuit structure (per layer):
#     q0: Ry(x) → Rz(θ₁) → Ry(θ₂) → Rz(θ₃) → Measure ⟨Z⟩
#
#     Key configurations:
#     - Number of qubits: 1
#     - Layers: 3 (paper's hyperparameter)
#     - Trainable parameters: 9 (3 rotation gates per layer × 3 layers)
#     - Data encoding: Ry rotation (data reuploading in each layer)
#     """
#
#     class QLayer(tq.QuantumModule):
#         def __init__(self):
#             super().__init__()
#             self.n_wires = 1
#             self.rz0 = tq.RZ(has_params=True, trainable=True)
#             self.ry0 = tq.RY(has_params=True, trainable=True)
#             self.rz1 = tq.RZ(has_params=True, trainable=True)
#
#         def forward(self, device: tq.QuantumDevice):
#             self.rz0(device, wires=0)
#             self.ry0(device, wires=0)
#             self.rz1(device, wires=0)
#
#     def __init__(self):
#         super().__init__()
#         self.n_wires = 1
#
#         self.encoder = tq.GeneralEncoder([
#             {'input_idx': [0], 'func': 'ry', 'wires': [0]},
#         ])
#
#         self.q_layer1 = self.QLayer()
#         self.q_layer2 = self.QLayer()
#         self.q_layer3 = self.QLayer()
#
#         self.measure = tq.MeasureAll(tq.PauliZ)
#
#     def forward(self, x, use_qiskit=False):
#         bsz = x.shape[0]
#         device = tq.QuantumDevice(n_wires=1, bsz=bsz, device=x.device)
#
#         self.encoder(device, x)
#         self.q_layer1(device)
#
#         self.encoder(device, x)
#         self.q_layer2(device)
#
#         self.encoder(device, x)
#         self.q_layer3(device)
#
#         measurement = self.measure(device)  # [batch*784, 1]
#         qout = (measurement.squeeze() + 1) / 2  # [batch*784]
#
#         return qout
#
#
# class JointMembership_FuzzyLayer(nn.Module):
#     """
#     Key features:
#     1. **No pooling**: Directly processes 28×28=784 pixels
#     2. **Pixel-wise processing**: Each pixel independently passes through a single-qubit circuit
#     3. **10 quantum circuits**: One single-qubit circuit per class
#     4. **Batch parallelism**: All pixels processed in one batch
#
#     Input: [batch, 1, 28, 28] - Raw images
#     Output: [batch, 784, num_classes] - Membership degree per pixel per class
#
#     Computational cost: 784 × num_classes quantum circuit calls/sample
#     """
#
#     def __init__(self, num_classes):
#         super(JointMembership_FuzzyLayer, self).__init__()
#         self.num_classes = num_classes
#         self.pixel_dim = 784  # 28×28
#
#         self.qfuzziers = nn.ModuleList([
#             SingleQubit_QuantumCircuit() for _ in range(num_classes)
#         ])
#         print(f"\n【Author's Architecture: Single-Qubit Fuzzy Layer】")
#         print(f"  - Input dimension: 28×28 = {self.pixel_dim} pixels")
#         print(f"  - Pooling strategy: No pooling (direct processing)")
#         print(f"  - Quantum circuits: {num_classes} single-qubit circuits")
#         print(f"  - Circuit layers: 3-layer data-reuploading")
#         print(f"  - Trainable parameters: 9/circuit × {num_classes} = {9 * num_classes} total")
#         print(f"  - Quantum circuit calls: {self.pixel_dim}×{num_classes}={self.pixel_dim * num_classes} calls/sample")
#         print(f"  - Output dimension: [batch, {self.pixel_dim}, {num_classes}]")
#         print(f"  - GPU parallelism: 100% (batch vectorization)\n")
#
#     def forward(self, x):
#         batch_size = x.size(0)
#
#         x_flat = x.view(batch_size, -1)  # [batch, 784]
#
#         x_all = x_flat.view(-1, 1)  # [batch*784, 1]
#
#         outputs = []
#         for qfuzzier in self.qfuzziers:
#             qout = qfuzzier(x_all)  # [batch*784]
#
#             qout = qout.view(batch_size, self.pixel_dim)
#             outputs.append(qout)
#
#         return torch.stack(outputs, dim=-1)
#
#     # ==================== Classical CNN feature extractor====================
#
#
# class classical_layer_dmnist(nn.Module):
#     """
#     Classic CNN feature extractor specialized for Dirty MNIST
#     Input: [batch, 1, 28, 28] (grayscale images)
#     Output: [batch, fusion_dim] (configurable feature dimension)
#
#     Architecture:
#     - Conv1: 28×28×1 → 12×12×10 (kernel=5, maxpool=2)
#     - Conv2: 12×12×10 → 4×4×20 (kernel=5, maxpool=2)
#     - FC: 320 → fusion_dim
#     """
#
#     def __init__(self, num_classes=10, fusion_dim=128):
#         super(classical_layer_dmnist, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=10,
#                                kernel_size=5, stride=1, padding=0)
#         self.maxpool1 = nn.MaxPool2d(2)
#
#         self.conv2 = nn.Conv2d(in_channels=10, out_channels=20,
#                                kernel_size=5, stride=1, padding=0)
#         self.maxpool2 = nn.MaxPool2d(2)
#
#         self.linear1 = nn.Linear(320, fusion_dim)
#
#         self.relu = nn.LeakyReLU()
#         self.dropout = nn.Dropout(p=0.3)
#
#     def forward(self, x):
#         x = self.relu(self.maxpool1(self.conv1(x)))  # [batch, 10, 12, 12]
#         x = self.relu(self.maxpool2(self.conv2(x)))  # [batch, 20, 4, 4]
#
#         x = x.view(x.size(0), -1)  # [batch, 320]
#
#         x = self.relu(self.linear1(x))  # [batch, fusion_dim]
#         x = self.dropout(x)
#
#         return x
#
#     # ==================== Full Network====================
#
#
# class DirtyMnist_JointMembership(nn.Module):
#     """
#     Quantum-Classical Hybrid Neural Network for Dirty MNIST (Author's Single-Qubit Architecture)
#
#     Architecture Components:
#     1. Classical branch: CNN feature extractor (specialized for grayscale images)
#     2. Quantum branch: Single-qubit fuzzy layer (no pooling, processes 784 pixels individually)
#     3. Fusion layer: Classical features + quantum features
#     4. Classifier: Fully-connected layer
#
#     Quantum Circuit Configuration (Author's Paper):
#     - **Single-qubit**: 1 quantum circuit per class
#     - **3-layer data-reuploading**: Data re-encoding per layer
#     - **No pooling**: Direct processing of 784 pixels
#     - **Pixel-wise processing**: Independent quantum computation per pixel
#
#     Key Dimension Transformations:
#     - Input: [batch, 1, 28, 28]
#     - Flatten: [batch, 784]
#     - Quantum output: [batch, 784, 10]
#     - Fuzzy rules: [batch, 10]
#     - Fusion: [batch, 128]
#     - Classification: [batch, 10]
#
#     Computational Cost:
#     - Quantum circuit calls/sample: 784 × 10 = 7,840
#     - Estimated training time: ~15-20 minutes/epoch (GPU-dependent)
#
#     Fuzzy Logic:
#     - Implements fuzzy "AND" using torch.prod
#     - Maintains original implementation from paper (no numerical stability tricks)
#     - ⚠️ Warning: Multiplying 784 values may cause numerical underflow and vanishing gradients
#
#     Parameters:
#     - num_classes: Number of classes (default 10)
#     - fusion_dim: Fusion layer dimension (recommended 128~256)
#     - n_random_pairs: Compatibility parameter (ignored in single-qubit architecture)
#     """
#
#     def __init__(self, num_classes=10, fusion_dim=128, n_random_pairs=0):
#         super(DirtyMnist_JointMembership, self).__init__()
#         self.k = num_classes
#
#         # ⚠️ n_random_pairs parameter is ignored (single-qubit architecture doesn't use pairing)
#         if n_random_pairs != 0:
#             print(f"  ⚠️  警告: n_random_pairs={n_random_pairs} 在单量子架构中被忽略")
#
#         # Classical CNN feature extraction (specialized for grayscale images)
#         self.class_layer = classical_layer_dmnist(num_classes, fusion_dim)
#         self.qfuzzy_layer = JointMembership_FuzzyLayer(num_classes=num_classes)
#
#         self.flinear = nn.Linear(self.k, fusion_dim)
#         self.classi = nn.Linear(fusion_dim, self.k)
#
#         print(f"\n【Dirty MNIST Single-Qubit Network Configuration (Author's Architecture)】")
#         print(f"  - Classical branch: CNN → {fusion_dim}D")
#         print(f"  - Quantum branch: Single-qubit×784 → {num_classes}D")
#         print(f"  - Fuzzy rules: torch.prod (original AND logic)")
#         print(f"  - Fusion method: torch.add")
#         print(f"  - Classifier: {fusion_dim} → {num_classes}")
#         print(f"  - Total computations: {784 * num_classes} quantum circuits/sample")
#         print(f"  - Expected accuracy: ~85% (paper benchmark)\n")
#
#     def forward(self, x):
#         batch_size = x.size(0)
#
#         c_part = self.class_layer(x)  # [batch, fusion_dim]
#
#         fuzzy_output = self.qfuzzy_layer(x)  # [batch, 784, 10]
#
#         fuzzy_rule_output = torch.prod(fuzzy_output, dim=1)  # [batch, 10]
#
#         fusion_output = torch.add(c_part, self.flinear(fuzzy_rule_output))
#
#         output = self.classi(fusion_output)
#         return output
#

# ============================resnet18=============================
"""
Dirty MNIST-specific ResNet18 Architecture (Pure Classical CNN Benchmark)
- Purpose: Classical benchmark for comparing quantum/fuzzy methods
- Architecture: Standard ResNet18 (adapted for single-channel input)
- Interface: Fully compatible with FDNN code, plug-and-play replacement for DirtyMnist_JointMembership
- Features: No fuzzy branch, pure classical deep learning approach
"""

import torch
import torch.nn as nn
import torchvision.models as models


class DirtyMnist_JointMembership(nn.Module):
    """
    Dirty MNIST-specific ResNet18 Model (Pure Classical CNN Benchmark)

    Architecture Description:
    - Based on torchvision.models.resnet18
    - Modified input channels: 3 → 1 (grayscale images)
    - Modified output dimension: 1000 classes → 10 classes (Dirty MNIST)
    - No fuzzy branch: Pure classical deep learning approach

    Interface Compatibility:
    - Class Name: DirtyMnist_JointMembership (identical to FDNN/QAHFNN)
    - Parameters: num_classes, fusion_dim, n_random_pairs (compatible but some ignored)
    - Input: [batch, 1, 28, 28]
    - Output: [batch, num_classes]

    Usage:
    ```python
    # Direct replacement for FDNN/QAHFNN, no main code modification needed
    model = DirtyMnist_JointMembership(num_classes=10)
    output = model(images)  # [batch, 10]
    ```

    Parameter Description:
    - num_classes: Number of classes (default=10, Dirty MNIST standard)
    - fusion_dim: Compatibility parameter (ignored in ResNet18, no fusion layer)
    - n_random_pairs: Compatibility parameter (ignored in ResNet18, no quantum pairing)

    Expected Performance:
    - Accuracy: ~85-88% (pure CNN benchmark)
    - Training Time: ~3-5 minutes/epoch
    - Parameter Count: ~11M (standard ResNet18)
    """

    def __init__(self, num_classes=10, fusion_dim=128, n_random_pairs=0):
        super(DirtyMnist_JointMembership, self).__init__()
        self.k = num_classes

        # ⚠️ fusion_dim and n_random_pairs parameters are ignored (ResNet18 has no fusion layer)
        if fusion_dim != 128:
            print(f"  ⚠️  警告: fusion_dim={fusion_dim} 在ResNet18架构中被忽略")
        if n_random_pairs != 0:
            print(f"  ⚠️  警告: n_random_pairs={n_random_pairs} 在ResNet18架构中被忽略")

        # Load pretrained ResNet18 backbone (without using pretrained weights)
        self.resnet18 = models.resnet18(weights=None)

        # Modify first conv layer: 3 channels → 1 channel (adapted for grayscale Dirty MNIST)
        self.resnet18.conv1 = nn.Conv2d(
            in_channels=1,  # 灰度图像
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, num_classes)

        print(f"\n【Dirty MNIST ResNet18 Configuration (Pure Classical CNN Benchmark)】")
        print(f"  - Architecture Type: ResNet18 (torchvision)")
        print(f"  - Input Adaptation: 3 channels → 1 channel (grayscale images)")
        print(f"  - Output Dimension: 1000 classes → {num_classes} classes")
        print(f"  - Feature Dimension: {num_ftrs}D (ResNet18 standard)")
        print(f"  - Fuzzy Branch: None (Pure Classical CNN)")
        print(f"  - Fusion Strategy: None (Direct classification)")
        print(f"  - Parameter Count: ~11M")
        print(f"  - Expected Accuracy: ~85-88%")
        print(f"  - Comparison Methods: FDNN/QAHFNN (Fuzzy/Quantum approaches)\n")

    def forward(self, x):
        x = self.resnet18.conv1(x)  # [batch, 64, 14, 14]
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)  # [batch, 64, 7, 7]

        x = self.resnet18.layer1(x)  # [batch, 64, 7, 7]
        x = self.resnet18.layer2(x)  # [batch, 128, 4, 4]
        x = self.resnet18.layer3(x)  # [batch, 256, 2, 2]
        x = self.resnet18.layer4(x)  # [batch, 512, 1, 1]

        x = self.resnet18.avgpool(x)  # [batch, 512, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 512]
        x = self.resnet18.fc(x)  # [batch, num_classes]

        return x

# =============================resnet50
# """
# Dirty MNIST-specific ResNet50 Architecture (Pure Classical CNN Benchmark)
# - Purpose: Classical benchmark for comparing quantum/fuzzy methods
# - Architecture: Standard ResNet50 (adapted for single-channel input)
# - Interface: Fully compatible with FDNN code, plug-and-play replacement for DirtyMnist_JointMembership
# - Features: No fuzzy branch, pure classical deep learning approach
# """
#
# import torch
# import torch.nn as nn
# import torchvision.models as models
#
#
# class DirtyMnist_JointMembership(nn.Module):
#     """
#     Dirty MNIST-specific ResNet50 Model (Pure Classical CNN Benchmark)
#
#     Architecture Description:
#     - Based on torchvision.models.resnet50
#     - Modified input channels: 3 → 1 (grayscale images)
#     - Modified output dimension: 1000 classes → 10 classes (Dirty MNIST)
#     - No fuzzy branch: Pure classical deep learning approach
#
#     Interface Compatibility:
#     - Class Name: DirtyMnist_JointMembership (identical to FDNN/QAHFNN)
#     - Parameters: num_classes, fusion_dim, n_random_pairs (compatible but some ignored)
#     - Input: [batch, 1, 28, 28]
#     - Output: [batch, num_classes]
#
#     Usage:
#     ```python
#     # Direct replacement for FDNN/QAHFNN, no main code modification needed
#     model = DirtyMnist_JointMembership(num_classes=10)
#     output = model(images)  # [batch, 10]
#     ```
#
#     Parameter Description:
#     - num_classes: Number of classes (default=10, Dirty MNIST standard)
#     - fusion_dim: Compatibility parameter (ignored in ResNet50, no fusion layer)
#     - n_random_pairs: Compatibility parameter (ignored in ResNet50, no quantum pairing)
#
#     Expected Performance:
#     - Accuracy: ~88-90% (deep CNN benchmark)
#     - Training Time: ~8-12 minutes/epoch (slower than ResNet18)
#     - Parameter Count: ~25M (standard ResNet50)
#     """
#
#     def __init__(self, num_classes=10, fusion_dim=128, n_random_pairs=0):
#         super(DirtyMnist_JointMembership, self).__init__()
#         self.k = num_classes
#
#         # ⚠️ fusion_dim and n_random_pairs parameters are ignored (ResNet50 has no fusion layer)
#         if fusion_dim != 128:
#             print(f"  ⚠️  警告: fusion_dim={fusion_dim} 在ResNet50架构中被忽略")
#         if n_random_pairs != 0:
#             print(f"  ⚠️  警告: n_random_pairs={n_random_pairs} 在ResNet50架构中被忽略")
#
#         # Load pretrained ResNet50 backbone (without using pretrained weights)
#         self.resnet50 = models.resnet50(weights=None)
#
#         # Modify first conv layer: 3 channels → 1 channel (adapted for grayscale Dirty MNIST)
#         self.resnet50.conv1 = nn.Conv2d(
#             in_channels=1,
#             out_channels=64,
#             kernel_size=7,
#             stride=2,
#             padding=3,
#             bias=False
#         )
#
#         # Modify fc layer: 1000 classes → num_classes classes
#         num_ftrs = self.resnet50.fc.in_features
#         self.resnet50.fc = nn.Linear(num_ftrs, num_classes)
#
#         print(f"\n【Dirty MNIST ResNet50 Configuration (Pure Classical CNN Benchmark)】")
#         print(f"  - Architecture Type: ResNet50 (torchvision)")
#         print(f"  - Input Adaptation: 3 channels → 1 channel (grayscale images)")
#         print(f"  - Output Dimension: 1000 classes → {num_classes} classes")
#         print(f"  - Feature Dimension: {num_ftrs}D (ResNet50 standard)")
#         print(f"  - Fuzzy Branch: None (Pure Classical CNN)")
#         print(f"  - Fusion Strategy: None (Direct classification)")
#         print(f"  - Parameter Count: ~25M")
#         print(f"  - Expected Accuracy: ~88-90%")
#         print(f"  - Comparison Methods: FDNN/QAHFNN (Fuzzy/Quantum approaches)\n")
#
#     def forward(self, x):
#
#
#         x = self.resnet50.conv1(x)  # [batch, 64, 14, 14]
#         x = self.resnet50.bn1(x)
#         x = self.resnet50.relu(x)
#         x = self.resnet50.maxpool(x)  # [batch, 64, 7, 7]
#
#         x = self.resnet50.layer1(x)  # [batch, 256, 7, 7]
#         x = self.resnet50.layer2(x)  # [batch, 512, 4, 4]
#         x = self.resnet50.layer3(x)  # [batch, 1024, 2, 2]
#         x = self.resnet50.layer4(x)  # [batch, 2048, 1, 1]
#
#         x = self.resnet50.avgpool(x)  # [batch, 2048, 1, 1]
#         x = x.view(x.size(0), -1)  # [batch, 2048]
#         x = self.resnet50.fc(x)  # [batch, num_classes]
#
#         return x
#
#
# #============================pure DNN-dirtymnist using================================
# """
# Dirty MNIST-specific DNN Benchmark (Pure Classical CNN Classifier)
# - Purpose: Reproduce DNN benchmark from paper Table I (83.6% accuracy)
# - Architecture: Pure CNN classifier (no fuzzy layer)
# - Processing: Standard CNN pipeline (Conv → Pooling → FC → Classification)
# - Class Name: Fully compatible with main program (plug-and-play replacement for QAHFNN/FDNN)
# """
#
# import torch
# import torch.nn as nn
#
#
# # ==================== Classical CNN Feature Extractor (Identical to QAHFNN/FDNN) ==============================
# class classical_layer_dmnist(nn.Module):
#     """
#     Dirty MNIST-specific Classical CNN Feature Extractor
#     Input: [batch, 1, 28, 28] (grayscale images)
#     Output: [batch, fusion_dim] (configurable feature dimension)
#
#     Architecture Description:
#     - Conv1: 28×28×1 → 12×12×10 (kernel=5, maxpool=2)
#     - Conv2: 12×12×10 → 4×4×20 (kernel=5, maxpool=2)
#     - FC: 320 → fusion_dim
#     """
#
#     def __init__(self, num_classes=10, fusion_dim=128):
#         super(classical_layer_dmnist, self).__init__()
#
#         # first conv layer: 28×28×1 → 24×24×10 → 12×12×10
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=10,
#                                kernel_size=5, stride=1, padding=0)
#         self.maxpool1 = nn.MaxPool2d(2)
#
#         # second conv layer : 12×12×10 → 8×8×20 → 4×4×20
#         self.conv2 = nn.Conv2d(in_channels=10, out_channels=20,
#                                kernel_size=5, stride=1, padding=0)
#         self.maxpool2 = nn.MaxPool2d(2)
#
#         # FC layer: 4×4×20=320 → fusion_dim
#         self.linear1 = nn.Linear(320, fusion_dim)
#
#         # activate func Dropout
#         self.relu = nn.LeakyReLU()
#         self.dropout = nn.Dropout(p=0.3)
#
#     def forward(self, x):
#
#         x = self.relu(self.maxpool1(self.conv1(x)))  # [batch, 10, 12, 12]
#         x = self.relu(self.maxpool2(self.conv2(x)))  # [batch, 20, 4, 4]
#
#
#         x = x.view(x.size(0), -1)  # [batch, 320]
#
#
#         x = self.relu(self.linear1(x))  # [batch, fusion_dim]
#         x = self.dropout(x)
#
#         return x
#
#
# # ==================== Pure DNN Classifier ====================
# class DirtyMnist_JointMembership(nn.Module):
#     """
#     Dirty MNIST-specific Pure DNN Classifier (Paper Table I Benchmark)
#
#     Architecture Components:
#     1. CNN Feature Extractor: Conv1→Pool1→Conv2→Pool2→FC
#     2. Classifier: Fully connected layer
#
#     **Key Differences**:
#     - No fuzzy layer (differs from FDNN/QAHFNN)
#     - No feature fusion (direct CNN→classification)
#     - Corresponds to "DNN" benchmark in paper Table I (83.6% accuracy)
#
#     Dimensional Flow:
#     - Input: [batch, 1, 28, 28]
#     - CNN Features: [batch, fusion_dim]
#     - Classification Output: [batch, num_classes]
#
#     Computational Cost:
#     - Standard CNN computation (no additional fuzzy calculations)
#     - Training Time: ~3-5 minutes/epoch (fastest)
#
#     Parameters:
#     - num_classes: Number of classes (default=10)
#     - fusion_dim: CNN feature dimension, recommended 128
#     - n_random_pairs: Compatibility parameter (ignored in DNN)
#
#     Interface Compatibility:
#     - ✅ Class name identical to QAHFNN/FDNN
#     - ✅ Parameter interface fully identical
#     - ✅ Input/output formats identical
#     - ✅ Direct plug-and-play replacement
#     """
#
#     def __init__(self, num_classes=10, fusion_dim=128, n_random_pairs=0):
#         super(DirtyMnist_JointMembership, self).__init__()
#         self.k = num_classes
#
#
#         if n_random_pairs != 0:
#             print(f"  ⚠️  warning: n_random_pairs={n_random_pairs}  in DNN be ignored")
#
#
#         self.class_layer = classical_layer_dmnist(num_classes, fusion_dim)
#
#
#         self.classi = nn.Linear(fusion_dim, self.k)
#
#         print(f"\n【Dirty MNIST Pure DNN Network Configuration (Paper Table I Benchmark)】")
#         print(f"  - Architecture: Pure CNN classifier (no fuzzy layer)")
#         print(f"  - CNN Features: Conv→Pool→Conv→Pool→FC → {fusion_dim}D")
#         print(f"  - Classifier: {fusion_dim} → {num_classes}")
#         print(f"  - Fuzzy Layer: ❌ None (differs from FDNN/QAHFNN)")
#         print(f"  - Feature Fusion: ❌ None (direct CNN→classification)")
#         print(f"  - Training Speed: Fastest (no additional fuzzy computations)")
#         print(f"  - Paper Accuracy: 83.6% (Table I)")
#         print(f"  - Comparison Methods: FDNN(84.5%), QAHFNN(85.0%)")
#         print(f"  - Interface Compatibility: ✅ Fully identical to QAHFNN/FDNN\n")
#     def forward(self, x):
#         c_part = self.class_layer(x)  # [batch, fusion_dim]
#         output = self.classi(c_part)  # [batch, num_classes]
#
#         return output
