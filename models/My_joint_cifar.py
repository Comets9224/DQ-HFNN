# =========================DQ-HFNN Dual-Qubit Single Layer================================
"""
CIFAR-10 Joint Membership Quantum Model - Enhanced 3×3 Grid Version
"""
import torch
import torch.nn as nn
import torchquantum as tq


# ==================== Classical CNN Feature Extractor ====================
class Block(nn.Module):

    def __init__(self, inchannel, outchannel, res=True, stride=1):
        super(Block, self).__init__()
        self.res = res
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False),
                nn.BatchNorm2d(outchannel),
            )
        else:
            self.shortcut = nn.Sequential()

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.left(x)
        if self.res:
            out += self.shortcut(x)
        out = self.relu(out)
        return out


class classical_layer(nn.Module):
    """
    CIFAR-10 Classical CNN Feature Extractor (Maintaining Original Architecture)

    Architecture: ResNet-style residual block stacking
    Input: [batch, 3, 32, 32]
    Output: [batch, 256] - Feature vector
    """

    def __init__(self, cfg=[64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'], res=True):
        super(classical_layer, self).__init__()
        self.res = res
        self.cfg = cfg
        self.inchannel = 3
        self.futures = self.make_layer()
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(4 * 512, 256),
        )

    def make_layer(self):
        layers = []
        for v in self.cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Block(self.inchannel, v, self.res))
                self.inchannel = v
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.futures(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


# ==================== Quantum Circuit: Joint Membership - Single Layer - With Entanglement (Same as DirtyMNIST) ====================
class JointMembership_QuantumCircuit(nn.Module):
    """
        Joint Membership Quantum Circuit Model (Identical to DirtyMNIST)
        - 2 qubits
        - 1 layer (single data encoding)
        - Contains entanglement gate (CNOT)
        - **Simultaneously measures both qubits**, outputs 2-dimensional joint membership values
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
            # Maintain order but only one CNOT        self.rz0(device, wires=0)
            self.cnot(device, wires=[0, 1])  # Only one CNOT -> weaker entanglement
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


# ==================== Quantum Fuzzy Layer: 3×3 Grid Partition Sampling ====================
class JointMembership_FuzzyLayer(nn.Module):

    def __init__(self, num_classes=10, n_random_pairs=460):
        super(JointMembership_FuzzyLayer, self).__init__()
        self.num_classes = num_classes

        self.img_size = 32
        self.n_channels = 3
        self.pixel_dim = 3072

        self.grid_size = 3
        self.block_sizes = [0, 10, 21, 32]

        self.total_pairs = n_random_pairs
        self.n_random_pairs = n_random_pairs
        self.n_fixed_pairs = 0

        self.register_buffer('block_pixels', self._compute_block_partition())

        self.register_buffer('inter_block_pairs', self._compute_inter_block_pairs())

        self.valid_pixels_per_block = self._precompute_valid_pixels()

        self.sampling_plan = self._setup_random_sampling_plan()

        self.qfuzziers = nn.ModuleList([
            JointMembership_QuantumCircuit() for _ in range(num_classes)
        ])

        print(f"\n{'=' * 70}")
        print(f"【CIFAR-10 3×3 Grid Quantum Fuzzy Layer (No Dimensionality Reduction - Batch Parallel Optimization)】")
        print(f"{'=' * 70}")
        print(f"  Input Configuration:")
        print(f"    - Image Size: 32×32×3 = 3072 pixels")
        print(f"    - Processing Strategy: No dimensionality reduction (direct raw pixel processing)")
        print(f"\n  3×3 Grid Configuration:")
        print(f"    - Partitions: 3×3 = 9 blocks/channels")
        print(f"    - Block Boundaries: {self.block_sizes} (adapted for 32×32)")
        self._print_block_info()
        print(f"\n  Pairing Strategy (Total Pairs={self.total_pairs}):")
        print(f"    - Fixed Pairs: {self.n_fixed_pairs} pairs (adjacent within blocks)")
        print(f"    - Random Pairs: {self.n_random_pairs} pairs (cross-block dynamic)")
        print(f"    - Coverage: {self.total_pairs * 2 / self.pixel_dim * 100:.1f}% pixels involved in pairing")
        print(f"\n  Quantum Computation:")
        print(f"    - Number of Quantum Circuits: {num_classes} (independent per class)")
        print(f"    - Circuit Calls/Sample: {self.total_pairs * num_classes} times")
        print(f"    - Output Dimension: [batch, {self.total_pairs * 2}, {num_classes}]")
        print(f"\n  ✅ Performance Optimizations:")
        print(f"    - Pre-computed valid pixel indices")
        print(f"    - Batch-parallel random sampling")
        print(f"    - Vectorized index operations")
        print(f"    - Pre-computed sampling plan")
        print(f"{'=' * 70}\n")

    def _print_block_info(self):

        print(f"    - Block Size Distribution (per channel):")
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
        print(f"    - Single-channel pixels: {total_pixels}")
        print(f"    - RGB total pixels: {total_pixels * 3}")

    def _compute_block_partition(self):
        """
        Precompute pixel indices for 3×3 grid partitioning (adapted for 32×32×3)

        Strategy: Treat 3072 pixels as flattened RGB image
        Pixel index = channel * 1024 + row * 32 + col

        Returns: [9*3, max_block_size] tensor (27 blocks: 9 spatial blocks × 3 channels)
        """
        block_pixels_list = []
        max_size = 0

        for channel in range(self.n_channels):
            for block_row in range(self.grid_size):
                for block_col in range(self.grid_size):
                    pixels = []
                    row_start = self.block_sizes[block_row]
                    row_end = self.block_sizes[block_row + 1]
                    col_start = self.block_sizes[block_col]
                    col_end = self.block_sizes[block_col + 1]

                    for r in range(row_start, row_end):
                        for c in range(col_start, col_end):
                            pixel_idx = channel * 1024 + r * self.img_size + c
                            pixels.append(pixel_idx)

                    block_pixels_list.append(pixels)
                    max_size = max(max_size, len(pixels))

        block_pixels_padded = []
        for pixels in block_pixels_list:
            padded = pixels + [-1] * (max_size - len(pixels))
            block_pixels_padded.append(padded)

        return torch.tensor(block_pixels_padded, dtype=torch.long)

    def _compute_inter_block_pairs(self):
        """
        Precompute candidate block pairs for cross-block pairing (adjacent + diagonal, per-channel independent)

        Each channel has 9 blocks:
        - Horizontal adjacent: 6 pairs
        - Vertical adjacent: 6 pairs
        - Diagonal pairs: 8 pairs
        Total: 20 pairs/channel × 3 channels = 60 pairs

        Returns: [60, 2] - block pair indices
        """
        inter_pairs = []

        for channel in range(self.n_channels):
            base_block = channel * 9

            for row in range(3):
                for col in range(2):
                    b1 = base_block + row * 3 + col
                    b2 = base_block + row * 3 + col + 1
                    inter_pairs.append([b1, b2])

            for row in range(2):
                for col in range(3):
                    b1 = base_block + row * 3 + col
                    b2 = base_block + (row + 1) * 3 + col
                    inter_pairs.append([b1, b2])

            center = base_block + 4
            diagonal_pairs = [
                [base_block + 0, center], [base_block + 1, center], [base_block + 2, center],
                [base_block + 3, center], [base_block + 5, center],
                [center, base_block + 6], [center, base_block + 7], [center, base_block + 8],
            ]
            inter_pairs.extend(diagonal_pairs)

        print(f"    - Cross-block candidate pairs: {len(inter_pairs)} pairs (3 channels × 20 pairs/channel)")
        return torch.tensor(inter_pairs, dtype=torch.long)

    def _precompute_valid_pixels(self):
        """
        Precompute valid pixel indices for each block (removing padding -1)

        Returns: List[Tensor] - Valid pixels for 27 blocks (9 blocks × 3 channels)
        """
        valid_pixels = []
        for block_id in range(27):  # 9 blocks × 3 channels
            pixels = self.block_pixels[block_id]
            valid = pixels[pixels != -1]
            valid_pixels.append(valid)

        print(f"    - ✅ Precomputed valid pixel indices: 27 blocks (9 spatial × 3 channels)")
        return valid_pixels

    def _setup_random_sampling_plan(self):
        """
        Precompute random sampling plan

        Returns: List[Tuple[int, int, int]] - (block1, block2, n_samples)
        """
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
        Batch-parallel random sampling (same optimization strategy as DirtyMNIST)

        Returns: [batch, n_random, 2] - pixel pair indices
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

    def forward(self, x):
        """
        Forward pass (no dimensionality reduction version)

        Input: x [batch, 3, 32, 32]
        Output: [batch, total_pairs*2, num_classes]
        """
        batch_size = x.size(0)

        x_flat = x.view(batch_size, -1)  # [batch, 3072]

        all_pairs_pixels = self._sample_inter_block_pairs_optimized(
            batch_size, x.device
        )  # [batch, total_pairs, 2]

        pair_pixels = torch.zeros(batch_size, self.total_pairs, 2, device=x.device)
        for b in range(batch_size):
            pair_pixels[b] = x_flat[b, all_pairs_pixels[b]]  # [total_pairs, 2]

        outputs = []
        for qfuzzier in self.qfuzziers:
            pairs_flat = pair_pixels.view(-1, 2)  # [batch*total_pairs, 2]
            qout = qfuzzier(pairs_flat)  # [batch*total_pairs, 2]
            qout = qout.view(batch_size, self.total_pairs, 2)  # [batch, total_pairs, 2]
            class_output = qout.view(batch_size, -1)  # [batch, total_pairs*2]
            outputs.append(class_output)

        return torch.stack(outputs, dim=-1)  # [batch, total_pairs*2, num_classes]


# ==================== Complete Network: CIFAR-10 ====================
class Cifar10_JointMembership(nn.Module):

    def __init__(self, num_classes=10, hidden_dim=256, n_random_pairs=460):
        super(Cifar10_JointMembership, self).__init__()
        self.k = num_classes

        self.class_layer = classical_layer()

        self.qfuzzy_layer = JointMembership_FuzzyLayer(
            num_classes=num_classes,
            n_random_pairs=n_random_pairs
        )

        self.flinear = nn.Linear(self.k, hidden_dim)
        self.classi = nn.Linear(hidden_dim, self.k)

        print(f"\n{'=' * 70}")
        print(f"【CIFAR-10 Joint Membership Network - Enhanced 3×3 Grid Version】")
        print(f"{'=' * 70}")
        print(f"  Architecture Configuration:")
        print(f"    - Classical Branch: Maintains original CNN architecture")
        print(f"    - Quantum Branch: 3×3 grid sampling (no dimensionality reduction)")
        print(f"    - Fusion Dimension: {hidden_dim}")
        print(f"    - Output Classes: {num_classes}")
        print(f"\n  Dimensional Flow:")
        print(f"    - Input: [batch, 3, 32, 32]")
        print(f"    - Quantum Output: [batch, {n_random_pairs * 2}, {num_classes}]")
        print(f"    - Fuzzy Rules: [batch, {num_classes}]")
        print(f"    - Fusion Layer: [batch, {hidden_dim}]")
        print(f"    - Final Output: [batch, {num_classes}]")
        print(f"\n  Model Scale:")
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        quantum_params = sum(p.numel() for p in self.qfuzzy_layer.parameters() if p.requires_grad)
        classical_params = sum(p.numel() for p in self.class_layer.parameters() if p.requires_grad)
        print(f"    - Total Parameters: {total_params:,}")
        print(f"    - Quantum Branch: {quantum_params:,} ({quantum_params / total_params * 100:.2f}%)")
        print(f"    - Classical Branch: {classical_params:,} ({classical_params / total_params * 100:.2f}%)")
        print(f"{'=' * 70}\n")

    def forward(self, x):
        batch_size = x.size(0)

        c_part = self.class_layer(x)  # [batch, 256]

        fuzzy_output = self.qfuzzy_layer(x)  # [batch, n_pairs*2, num_classes]

        fuzzy_output_clamped = torch.clamp(fuzzy_output, min=1e-7, max=1.0)
        log_membership = torch.log(fuzzy_output_clamped)
        fuzzy_rule_output = torch.mean(log_membership, dim=1)  # [batch, num_classes]

        fusion_output = torch.add(c_part, self.flinear(fuzzy_rule_output))

        output = self.classi(fusion_output)
        return output

        #
        # #================================DNN===================================
        #
        # """
        # CIFAR-10 Pure Classical DNN Baseline Model
        # - Class Name: Cifar10_JointMembership (consistent with DQHFNN for plug-and-play compatibility)
        # - Interface: Fully compatible with DQHFNN calling conventions
        # - Architecture: Shares the same classical CNN branch, removes quantum fuzzy layer
        # """
        #
        # import torch
        # import torch.nn as nn
        #
        #
        # ==================== Classical CNN Feature Extractor (Identical to DQHFNN) ====================
        # class Block(nn.Module):
        #
        #
        #     def __init__(self, inchannel, outchannel, res=True, stride=1):
        #         super(Block, self).__init__()
        #         self.res = res
        #         self.left = nn.Sequential(
        #             nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, stride=stride, bias=False),
        #             nn.BatchNorm2d(outchannel),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1, stride=1, bias=False),
        #             nn.BatchNorm2d(outchannel),
        #         )
        #         if stride != 1 or inchannel != outchannel:
        #             self.shortcut = nn.Sequential(
        #                 nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False),
        #                 nn.BatchNorm2d(outchannel),
        #             )
        #         else:
        #             self.shortcut = nn.Sequential()
        #
        #         self.relu = nn.Sequential(
        #             nn.ReLU(inplace=True),
        #         )
        #
        #     def forward(self, x):
        #         out = self.left(x)
        #         if self.res:
        #             out += self.shortcut(x)
        #         out = self.relu(out)
        #         return out
        #
        #
        # class classical_layer(nn.Module):
        # """
        # CIFAR-10 Classical CNN Feature Extractor (Shared with DQHFNN)
        #
        # Architecture: ResNet-style residual block stacking
        # Input: [batch, 3, 32, 32]
        # Output: [batch, 256] - Feature vector
        # """
        #
        #     def __init__(self, cfg=[64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'], res=True):
        #         super(classical_layer, self).__init__()
        #         self.res = res
        #         self.cfg = cfg
        #         self.inchannel = 3
        #         self.futures = self.make_layer()
        #         self.classifier = nn.Sequential(
        #             nn.Dropout(0.4),
        #             nn.Linear(4 * 512, 256),
        #         )
        #
        #     def make_layer(self):
        #         layers = []
        #         for v in self.cfg:
        #             if v == 'M':
        #                 layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        #             else:
        #                 layers.append(Block(self.inchannel, v, self.res))
        #                 self.inchannel = v
        #         return nn.Sequential(*layers)
        #
        #     def forward(self, x):
        #         out = self.futures(x)
        #         out = out.view(out.size(0), -1)
        #         out = self.classifier(out)
        #         return out
        #
        #
        # ==================== Pure Classical DNN Baseline Model ====================
        # class Cifar10_JointMembership(nn.Module):
        #     """
        #     CIFAR-10 Pure Classical DNN Baseline Model (Plug-and-Play Version)
        #
        #     ⚠️ Note:
        #     - Class name intentionally matches DQHFNN (Cifar10_JointMembership)
        #     - Accepts same parameters (num_classes, hidden_dim, n_random_pairs)
        #     - n_random_pairs parameter is ignored (only for interface compatibility)
        #
        #     Architecture: Classical CNN feature extraction + classification layer
        #     - Exactly matches DQHFNN's classical branch
        #     - Removes quantum fuzzy layer, directly classifies
        #
        #     Parameters:
        #         num_classes: Number of classes (default=10)
        #         hidden_dim: Feature dimension (default=256)
        #         n_random_pairs: Quantum pairing count (invalid in DNN, only for interface compatibility)
        #     """
        #
        #     def __init__(self, num_classes=10, hidden_dim=256, n_random_pairs=460):
        #         super(Cifar10_JointMembership, self).__init__()
        #         self.k = num_classes
        #
        #         # Classical CNN feature extraction (identical to DQHFNN)
        #         self.class_layer = classical_layer()
        #
        #
        #         self.classifier = nn.Linear(hidden_dim, self.k)
        #
        # print(f"\n{'=' * 70}")
        # print(f"【CIFAR-10 Pure Classical DNN Baseline Model (Plug-and-Play Version)】")
        # print(f"{'=' * 70}")
        # print(f"  ⚠️  Model Type: Pure Classical DNN (No Quantum Branch)")
        # print(f"  ✅ Interface Compatibility: Fully consistent with DQHFNN")
        # print(f"\n  Architecture Configuration:")
        # print(f"    - Classical Branch: ResNet-style CNN (Same as DQHFNN)")
        # print(f"    - Quantum Branch: ❌ None (Removed)")
        # print(f"    - Feature Dimension: {hidden_dim}")
        # print(f"    - Output Classes: {num_classes}")
        # print(f"    - n_random_pairs: {n_random_pairs} (Parameter ignored, only for interface compatibility)")
        # print(f"\n  Dimensional Flow:")
        # print(f"    - Input: [batch, 3, 32, 32]")
        # print(f"    - CNN Features: [batch, {hidden_dim}]")
        # print(f"    - Final Output: [batch, {num_classes}]")
        # print(f"\n  Model Scale:")
        #         total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        #         cnn_params = sum(p.numel() for p in self.class_layer.parameters() if p.requires_grad)
        #         classifier_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        # print(f"    - Total Parameters: {total_params:,}")
        # print(f"    - CNN Feature Extraction: {cnn_params:,} ({cnn_params / total_params * 100:.2f}%)")
        # print(f"    - Classification Layer: {classifier_params:,} ({classifier_params / total_params * 100:.2f}%)")
        #         print(f"{'=' * 70}\n")
        #
        #     def forward(self, x):

        #         features = self.class_layer(x)  # [batch, 256]
        #         output = self.classifier(features)  # [batch, 10]
        #
        #         return output

        # =====================QAHFNN=================================
        # """
        # CIFAR-10 Single-Qubit QAHFNN Architecture
        # - Classical Branch: Maintains DQHFNN's ResNet-style CNN (completely unchanged)
        # - Quantum Branch: Single-qubit + 3-layer data-reuploading (migrated from Dirty MNIST)
        # - Processing Strategy: Processes 3072 pixels individually (no pooling)
        # - Class Name: Fully compatible with main program naming conventions (plug-and-play)
        # """
        # #
        # import torch
        # import torch.nn as nn
        # import torchquantum as tq
        #
        #
        # # ==================== 单量子比特电路（3层 data-reuploading）====================
        # class SingleQubit_QuantumCircuit(nn.Module):
        #     """
        #     Single-qubit circuit from the paper (3-layer data-reuploading)
        #
        #     Circuit structure (per layer):
        #     q0: Ry(x) → Rz(θ₁) → Ry(θ₂) → Rz(θ₃) → Measure ⟨Z⟩
        #
        #     Key configurations:
        #     - Number of qubits: 1
        #     - Layers: 3 (paper hyperparameter)
        #     - Trainable parameters: 9 (3 rotation gates per layer × 3 layers)
        #     - Data encoding: Ry rotation (data reuploaded each layer)
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
        #         self.encoder = tq.GeneralEncoder([
        #             {'input_idx': [0], 'func': 'ry', 'wires': [0]},
        #         ])
        #
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

        #         measurement = self.measure(device)  # [batch*3072, 1]
        #         qout = (measurement.squeeze() + 1) / 2  # [batch*3072]
        #
        #         return qout
        #
        #
        # # ==================== 量子模糊层（单量子比特版本 - CIFAR-10适配）====================
        # class JointMembership_FuzzyLayer(nn.Module):
        # """
        # CIFAR-10 Single-Qubit Fuzzy Layer
        #
        # Key Features:
        # 1. **No Pooling**: Directly processes 32×32×3=3072 pixels
        # 2. **Pixel-wise Processing**: Each pixel independently through single-qubit circuit
        # 3. **10 Quantum Circuits**: One single-qubit circuit per class
        # 4. **Batch Parallelism**: All pixels processed in one batch
        #
        # Input: [batch, 3, 32, 32] - RGB images
        # Output: [batch, 3072, num_classes] - Membership degree per pixel per class
        #
        # Computational Cost: 3072 × num_classes quantum circuit calls/sample
        # """
        #
        #     def __init__(self, num_classes):
        #         super(JointMembership_FuzzyLayer, self).__init__()
        #         self.num_classes = num_classes
        #         self.pixel_dim = 3072  # 32×32×3
        #         self.qfuzziers = nn.ModuleList([
        #             SingleQubit_QuantumCircuit() for _ in range(num_classes)
        #         ])
        #
        # print(f"\n{'=' * 70}")
        # print(f"【CIFAR-10 Single-Qubit Fuzzy Layer】")
        # print(f"{'=' * 70}")
        # print(f"  Quantum Circuit Configuration:")
        # print(f"    - Input Size: 32×32×3 = {self.pixel_dim} pixels")
        # print(f"    - Pooling Strategy: No pooling (direct processing)")
        # print(f"    - Quantum Circuits: {num_classes} single-qubit circuits")
        # print(f"    - Circuit Layers: 3-layer data-reuploading")
        # print(f"    - Trainable Parameters: 9/circuit × {num_classes} = {9 * num_classes}")
        # print(f"\n  Computational Cost:")
        # print(f"    - Quantum Circuit Calls/Sample: {self.pixel_dim} × {num_classes} = {self.pixel_dim * num_classes}")
        # print(f"    - Output Dimension: [batch, {self.pixel_dim}, {num_classes}]")
        # print(f"    - GPU Parallelism: 100% (batch vectorization)")
        # print(f"\n  Estimated Performance:")
        # print(f"    - Training Time: ~20-30 minutes/epoch (GPU dependent)")
        # print(f"    - Compared to Dirty MNIST: Computation ×3.9 (3072/784)")
        # print(f"{'=' * 70}\n")
        #     def forward(self, x):
        #         batch_size = x.size(0)
        #         outputs = []
        #         for qfuzzier in self.qfuzziers:
        #             qout = qfuzzier(x_all)  # [batch*3072]
        #             qout = qout.view(batch_size, self.pixel_dim)
        #             outputs.append(qout)
        #         return torch.stack(outputs, dim=-1)
        #
        #
        # ==================== Classical CNN Feature Extractor (Identical to DQHFNN) ====================
        # class Block(nn.Module):
        #     def __init__(self, inchannel, outchannel, res=True, stride=1):
        #         super(Block, self).__init__()
        #         self.res = res
        #         self.left = nn.Sequential(
        #             nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, stride=stride, bias=False),
        #             nn.BatchNorm2d(outchannel),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1, stride=1, bias=False),
        #             nn.BatchNorm2d(outchannel),
        #         )
        #         if stride != 1 or inchannel != outchannel:
        #             self.shortcut = nn.Sequential(
        #                 nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False),
        #                 nn.BatchNorm2d(outchannel),
        #             )
        #         else:
        #             self.shortcut = nn.Sequential()
        #
        #         self.relu = nn.Sequential(
        #             nn.ReLU(inplace=True),
        #         )
        #
        #     def forward(self, x):
        #         out = self.left(x)
        #         if self.res:
        #             out += self.shortcut(x)
        #         out = self.relu(out)
        #         return out
        #
        #
        # class classical_layer(nn.Module):
        #
        #     def __init__(self, cfg=[64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'], res=True):
        #         super(classical_layer, self).__init__()
        #         self.res = res
        #         self.cfg = cfg
        #         self.inchannel = 3
        #         self.futures = self.make_layer()
        #         self.classifier = nn.Sequential(
        #             nn.Dropout(0.4),
        #             nn.Linear(4 * 512, 256),
        #         )
        #
        #     def make_layer(self):
        #         layers = []
        #         for v in self.cfg:
        #             if v == 'M':
        #                 layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        #             else:
        #                 layers.append(Block(self.inchannel, v, self.res))
        #                 self.inchannel = v
        #         return nn.Sequential(*layers)
        #
        #     def forward(self, x):
        #         out = self.futures(x)
        #         out = out.view(out.size(0), -1)
        #         out = self.classifier(out)
        #         return out
        #
        #
        # ==================== Complete QAHFNN Network (Single-Qubit Version) ====================
        # class Cifar10_JointMembership(nn.Module):
        #     """
        #     CIFAR-10 Single-Qubit QAHFNN Architecture
        #
        #     Architecture Components:
        #     1. Classical Branch: ResNet-style CNN (identical to DQHFNN)
        #     2. Quantum Branch: Single-qubit fuzzy layer (no pooling, processes 3072 pixels individually)
        #     3. Fusion Layer: Classical features + quantum features
        #     4. Classifier: Fully connected layer
        #
        #     Quantum Circuit Configuration:
        #     - **Single-qubit**: One quantum circuit per class
        #     - **3-layer data-reuploading**: Data re-encoded each layer
        #     - **No pooling**: Directly processes 3072 pixels
        #     - **Pixel-wise processing**: Independent quantum computation per pixel
        #
        #     Key Dimensional Flow:
        #     - Input: [batch, 3, 32, 32]
        #     - Flattened: [batch, 3072]
        #     - Quantum Output: [batch, 3072, 10]
        #     - Fuzzy Rules: [batch, 10]
        #     - Fusion: [batch, 256]
        #     - Classification: [batch, 10]
        #
        #     Computational Cost:
        #     - Quantum circuit calls/sample: 3072 × 10 = 30720
        #     - Estimated training time: ~20-30 minutes/epoch (GPU dependent)
        #
        #     Fuzzy Logic:
        #     - Uses torch.prod to implement fuzzy "AND" logic
        #     - Maintains original implementation from author's paper
        #     - ⚠️ Note: Multiplying 3072 values may cause numerical underflow and gradient vanishing
        #
        #     Parameters:
        #     - num_classes: Number of classes (default=10)
        #     - hidden_dim: Fusion layer dimension (default=256, consistent with DQHFNN)
        #     - n_random_pairs: Compatibility parameter (ignored in single-qubit architecture)
        #     """
        #
        #     def __init__(self, num_classes=10, hidden_dim=256, n_random_pairs=0):
        #         super(Cifar10_JointMembership, self).__init__()
        #         self.k = num_classes
        #
        #         # ⚠️ n_random_pairs参数被忽略（单量子架构不使用配对）
        #         if n_random_pairs != 0:
        #             print(f"  ⚠️  Warning: n_random_pairs={n_random_pairs} is ignored in single-qubit architecture")
        #         self.class_layer = classical_layer()
        #
        #         self.qfuzzy_layer = JointMembership_FuzzyLayer(num_classes=num_classes)
        #
        #         self.flinear = nn.Linear(self.k, hidden_dim)
        #         self.classi = nn.Linear(hidden_dim, self.k)
        #
        # print(f"\n{'=' * 70}")
        # print(f"【CIFAR-10 Single-Qubit QAHFNN Network Configuration】")
        # print(f"{'=' * 70}")
        # print(f"  Architecture Configuration:")
        # print(f"    - Classical Branch: ResNet-style CNN → {hidden_dim}D")
        # print(f"    - Quantum Branch: Single-qubit×3072 → {num_classes}D")
        # print(f"    - Fuzzy Rules: torch.prod (original AND logic)")
        # print(f"    - Fusion Method: torch.add")
        # print(f"    - Classifier: {hidden_dim} → {num_classes}")
        # print(f"\n  Dimensional Flow:")
        # print(f"    - Input: [batch, 3, 32, 32]")
        # print(f"    - CNN Features: [batch, {hidden_dim}]")
        # print(f"    - Quantum Output: [batch, 3072, {num_classes}]")
        # print(f"    - Fuzzy Rules: [batch, {num_classes}]")
        # print(f"    - Fusion Layer: [batch, {hidden_dim}]")
        # print(f"    - Final Output: [batch, {num_classes}]")
        # print(f"\n  Model Scale:")
        #         total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        #         quantum_params = sum(p.numel() for p in self.qfuzzy_layer.parameters() if p.requires_grad)
        #         classical_params = sum(p.numel() for p in self.class_layer.parameters() if p.requires_grad)
        #         print(f"    - Total Parameters: {total_params:,}")
        #         print(f"    - Quantum Branch: {quantum_params:,} ({quantum_params / total_params * 100:.2f}%)")
        #         print(f"    - Classical Branch: {classical_params:,} ({classical_params / total_params * 100:.2f}%)")
        #         print(f"\n  Computational Cost:")
        #         print(f"    - Quantum Circuit Calls/Sample: 3072 × {num_classes} = {3072 * num_classes}")
        #         print(f"    - Compared to Dirty MNIST: Computation×3.9")
        #         print(f"\n  Comparative Experiments:")
        #         print(f"    - Baseline Model: DNN (Pure Classical)")
        #         print(f"    - Classical Fuzzy: FDNN (Gaussian Membership)")
        #         print(f"    - Dual-Qubit: DQHFNN (Dual-Qubit + Entanglement)")
        #         print(f"    - Single-Qubit: QAHFNN (This Model)")
        #         print(f"{'=' * 70}\n")
        #
        #     def forward(self, x):
        #         batch_size = x.size(0)
        #         c_part = self.class_layer(x)  # [batch, 256]
        #         fuzzy_output = self.qfuzzy_layer(x)  # [batch, 3072, 10]
        #         fuzzy_rule_output = torch.prod(fuzzy_output, dim=1)  # [batch, 10]
        #         fusion_output = torch.add(c_part, self.flinear(fuzzy_rule_output))
        #         output = self.classi(fusion_output)
        #         return output

        # == == == == == == == == == == =FDNN == == == == == == == == == == == == == == == == ==
        # """
        # CIFAR-10 FDNN Architecture (Classical Gaussian Fuzzy Function)
        # - Classical Branch: Maintains DQHFNN's ResNet-style CNN (completely unchanged)
        # - Fuzzy Branch: Gaussian membership function (replaces single-qubit circuit)
        # - Processing Strategy: Processes 3072 pixels individually (no pooling)
        # - Pixel-level independent parameters: Each pixel has independent mean and std per class
        # - Class Name: Fully compatible with main program naming conventions (plug-and-play)
        # """
        # import torch
        # import torch.nn as nn
        #
        #
        # ==================== Classical Fuzzy Layer (Replaces Quantum Layer) ====================
        # class JointMembership_FuzzyLayer(nn.Module):
        #     """
        #     CIFAR-10 Classical Gaussian Fuzzy Layer (Replaces Single-Qubit Circuit)
        #
        #     Key Features:
        #     1. **No Pooling**: Directly processes 32×32×3=3072 pixels
        #     2. **Pixel-wise Processing**: Each pixel independently through Gaussian membership function
        #     3. **Pixel-level Independent Parameters**: Independent mean and std per pixel per class
        #     4. **Batch Parallelism**: All pixels processed in one batch
        #
        #     Input: [batch, 3, 32, 32] - RGB images
        #     Output: [batch, 3072, num_classes] - Membership degree per pixel per class
        #
        #     Parameter Count: 3072 pixels × num_classes × 2 parameters (mean+std) = 61,440 parameters (for 10 classes)
        #     """
        #
        #     def __init__(self, num_classes):
        #         super(JointMembership_FuzzyLayer, self).__init__()
        #         self.num_classes = num_classes
        #         self.pixel_dim = 3072  # 32×32×3
        #
        #         #Shape: [3072, num_classes] - Independent parameters per pixel per class
        #         self.means = nn.Parameter(torch.randn(self.pixel_dim, num_classes) * 0.5)
        #         self.stds = nn.Parameter(torch.ones(self.pixel_dim, num_classes) * 0.5)
#
#         print(f"\n{'=' * 70}")
#         print(f"【CIFAR-10 Classical Gaussian Fuzzy Layer - Pixel-level Independent Parameters】")
#         print(f"{'=' * 70}")
#         print(f"  Fuzzy Function Configuration:")
#         print(f"    - Input Size: 32×32×3 = {self.pixel_dim} pixels")
#         print(f"    - Pooling Strategy: No pooling (direct processing)")
#         print(f"    - Parameter Granularity: Pixel-level independent parameters (matches paper design)")
#         print(f"    - Per pixel per class: Independent mean + std")
#         print(f"\n  Parameter Scale:")
#         print(f"    - Means parameters: [{self.pixel_dim}, {num_classes}]")
#         print(f"    - Stds parameters: [{self.pixel_dim}, {num_classes}]")
#         print(f"    - Total Parameters: {self.pixel_dim} × {num_classes} × 2 = {self.pixel_dim * num_classes * 2:,}")
#         print(f"\n  Computation Flow:")
#         print(f"    - Input Flattening: [batch, 3, 32, 32] → [batch, 3072]")
#         print(f"    - Gaussian Computation: Independent membership degree per pixel per class")
#         print(f"    - Output Dimension: [batch, {self.pixel_dim}, {num_classes}]")
#         print(f"    - GPU Parallelism: 100% (batch vectorization)")
#         print(f"\n  Comparison:")
#         print(f"    - Class-level Parameters (Incorrect): {num_classes} × 2 = {num_classes * 2}")
#         print(f"    - Pixel-level Parameters (Correct): {self.pixel_dim * num_classes * 2:,}")
#         print(f"    - Parameter Multiplier: {(self.pixel_dim * num_classes * 2) / (num_classes * 2):.0f}x")
#         print(f"{'=' * 70}\n")
#
#
# def forward(self, x):
#     batch_size = x.size(0)
#
#     x_flat = x.view(batch_size, -1)  # [batch, 3072]
#
#     # [batch, 3072] → [batch, 3072, 1]
#     x_expanded = x_flat.unsqueeze(-1)  # [batch, 3072, 1]
#
#     # x_expanded: [batch, 3072, 1]
#     # self.means: [3072, num_classes]
#     # self.stds: [3072, num_classes]
#     # 广播计算: [batch, 3072, 1] - [3072, num_classes] → [batch, 3072, num_classes]
#
#     membership = torch.exp(
#         -((x_expanded - self.means) ** 2) / (2 * self.stds ** 2 + 1e-8)
#     )  # [batch, 3072, num_classes]
#
#     return membership
#
#
# # ==================== Classical CNN Feature Extractor (Identical to DQHFNN) ====================
# class Block(nn.Module):
#
#     def __init__(self, inchannel, outchannel, res=True, stride=1):
#         super(Block, self).__init__()
#         self.res = res
#         self.left = nn.Sequential(
#             nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, stride=stride, bias=False),
#             nn.BatchNorm2d(outchannel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1, stride=1, bias=False),
#             nn.BatchNorm2d(outchannel),
#         )
#         if stride != 1 or inchannel != outchannel:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(outchannel),
#             )
#         else:
#             self.shortcut = nn.Sequential()
#
#         self.relu = nn.Sequential(
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         out = self.left(x)
#         if self.res:
#             out += self.shortcut(x)
#         out = self.relu(out)
#         return out
#
#
# class classical_layer(nn.Module):
#
#     def __init__(self, cfg=[64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'], res=True):
#         super(classical_layer, self).__init__()
#         self.res = res
#         self.cfg = cfg
#         self.inchannel = 3
#         self.futures = self.make_layer()
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.4),
#             nn.Linear(4 * 512, 256),
#         )
#
#     def make_layer(self):
#         layers = []
#         for v in self.cfg:
#             if v == 'M':
#                 layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
#             else:
#                 layers.append(Block(self.inchannel, v, self.res))
#                 self.inchannel = v
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.futures(x)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         return out
#
#
# # ==================== Complete FDNN Network (Classical Gaussian Fuzzy Version - Pixel-level Independent Parameters) ====================
# class Cifar10_JointMembership(nn.Module):
#     """
#     CIFAR-10 Classical Fuzzy Deep Neural Network (FDNN - Gaussian Membership Function - Pixel-level Independent Parameters)
#
#     Architecture Components:
#     1. Classical Branch: ResNet-style CNN (identical to DQHFNN)
#     2. Classical Fuzzy Branch: Gaussian membership function (pixel-level independent parameters, no pooling, processes 3072 pixels individually)
#     3. Fusion Layer: Classical features + fuzzy features
#     4. Classifier: Fully connected layer
#
#     Fuzzy Configuration (Classical Method - Pixel-level Independent Parameters):
#     - **Gaussian Function**: Independent Gaussian membership function per pixel per class
#     - **Trainable Parameters**: Independent mean + std per pixel per class
#     - **Total Parameters**: 3072 pixels × 10 classes × 2 parameters = 61,440 fuzzy parameters
#     - **No Pooling**: Directly processes 3072 pixels
#     - **Pixel-wise Processing**: Independent membership calculation per pixel
#
#     Key Dimensional Flow:
#     - Input: [batch, 3, 32, 32]
#     - Flattened: [batch, 3072]
#     - Fuzzy Output: [batch, 3072, 10]
#     - Fuzzy Rules: [batch, 10]
#     - Fusion: [batch, 256]
#     - Classification: [batch, 10]
#
#     Computational Cost:
#     - Gaussian function calculations/sample: 3072 × 10 = 30720
#     - Estimated training time: ~5-10 minutes/epoch (faster than quantum approach)
#
#     Fuzzy Logic:
#     - Uses torch.prod to implement fuzzy "AND" logic
#     - Maintains consistency with QAHFNN original implementation
#     - ⚠️ Note: Multiplying 3072 values may cause numerical underflow and gradient vanishing
#
#     Parameters:
#     - num_classes: Number of classes (default=10)
#     - hidden_dim: Fusion layer dimension (default=256, consistent with DQHFNN)
#     - n_random_pairs: Compatibility parameter (ignored in FDNN)
#     """
#
#     def __init__(self, num_classes=10, hidden_dim=256, n_random_pairs=0):
#         super(Cifar10_JointMembership, self).__init__()
#         self.k = num_classes
#
#         # ⚠️ n_random_pairs parameter is ignored (not used in FDNN architecture)
#         if n_random_pairs != 0:
#             print(f"  ⚠️  警告: n_random_pairs={n_random_pairs} 在FDNN架构中被忽略")
#
#         # Classical CNN feature extraction (identical to DQHFNN)
#         self.class_layer = classical_layer()
#
#         # Classical Gaussian Fuzzy Layer (Replaces Quantum Layer - Pixel-level Independent Parameters)
#         self.qfuzzy_layer = JointMembership_FuzzyLayer(num_classes=num_classes)
#
#         # Fusion and Classification Layers
#         self.flinear = nn.Linear(self.k, hidden_dim)
#         self.classi = nn.Linear(hidden_dim, self.k)
#
#         print(f"\n{'=' * 70}")
#         print(f"【CIFAR-10 FDNN Network Configuration (Classical Gaussian Fuzzy - Pixel-level Independent Parameters)】")
#         print(f"{'=' * 70}")
#         print(f"  Architecture Configuration:")
#         print(f"    - Classical Branch: ResNet-style CNN → {hidden_dim}D")
#         print(f"    - Fuzzy Branch: Pixel-level Gaussian functions×3072 → {num_classes}D")
#         print(f"    - Membership Function: Gaussian (Pixel-level independent mean + std)")
#         print(f"    - Fuzzy Rules: torch.prod (Original AND logic)")
#         print(f"    - Fusion Method: torch.add")
#         print(f"    - Classifier: {hidden_dim} → {num_classes}")
#         print(f"\n  Dimensional Flow:")
#         print(f"    - Input: [batch, 3, 32, 32]")
#         print(f"    - CNN Features: [batch, {hidden_dim}]")
#         print(f"    - Fuzzy Output: [batch, 3072, {num_classes}]")
#         print(f"    - Fuzzy Rules: [batch, {num_classes}]")
#         print(f"    - Fusion Layer: [batch, {hidden_dim}]")
#         print(f"    - Final Output: [batch, {num_classes}]")
#         print(f"\n  Model Scale:")
#         total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         fuzzy_params = sum(p.numel() for p in self.qfuzzy_layer.parameters() if p.requires_grad)
#         classical_params = sum(p.numel() for p in self.class_layer.parameters() if p.requires_grad)
#         print(f"    - Total Parameters: {total_params:,}")
#         print(f"    - Fuzzy Branch: {fuzzy_params:,} ({fuzzy_params / total_params * 100:.2f}%)")
#         print(f"    - Classical Branch: {classical_params:,} ({classical_params / total_params * 100:.2f}%)")
#         print(f"\n  Fuzzy Layer Parameters:")
#         print(f"    - Pixel-level means: [3072, {num_classes}] = {3072 * num_classes:,}")
#         print(f"    - Pixel-level stds: [3072, {num_classes}] = {3072 * num_classes:,}")
#         print(f"    - Total Fuzzy Parameters: {3072 * num_classes * 2:,} (matches paper design)")
#         print(f"\n  Computational Cost:")
#         print(f"    - Gaussian Calculations/Sample: 3072 × {num_classes} = {3072 * num_classes}")
#         print(f"    - Compared to Dirty MNIST: Computation×3.9")
#         print(f"\n  Comparative Experiments:")
#         print(f"    - Baseline Model: DNN (Pure Classical)")
#         print(f"    - Classical Fuzzy: FDNN (This Model - Pixel-level Independent Parameters)")
#         print(f"    - Single-Qubit: QAHFNN (Single-Qubit)")
#         print(f"    - Dual-Qubit: DQHFNN (Dual-Qubit + Entanglement)")
#         print(f"\n  Expected Performance:")
#         print(f"    - Training Speed: Faster than quantum approaches")
#         print(f"    - Expected Accuracy: Serves as classical benchmark")
#         print(f"    - Parameter Count: Significantly higher than class-level parameter version")
#         print(f"{'=' * 70}\n")
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         c_part = self.class_layer(x)  # [batch, 256]
#         fuzzy_output = self.qfuzzy_layer(x)  # [batch, 3072, 10]
#         fuzzy_rule_output = torch.prod(fuzzy_output, dim=1)  # [batch, 10]
#         fusion_output = torch.add(c_part, self.flinear(fuzzy_rule_output))
#         output = self.classi(fusion_output)
#         return output
