"""
JAFFE Dual-Qubit Joint Membership Hybrid Neural Network (Unrestricted Version - For Regularization Experiments)

Removed range restrictions for n_random_pairs (allows 0-512)
Retains complete 3×3 grid partitioning logic
Uses new quantum circuit (single-CNOT version)
Cleans up all mutual information analysis interfaces (not needed for experiments)
"""

import torch
import torch.nn as nn
import torchquantum as tq


# ==================== Quantum Circuit: Joint Membership (New Version - Single CNOT) ====================
class JointMembership_QuantumCircuit(nn.Module):
    """
    Joint Membership Quantum Circuit Model (New Version - Weak Entanglement)
    - 2 qubits
    - 1 data encoding layer
    - Single CNOT gate (weak entanglement)
    - Simultaneous measurement of both qubits

    Circuit Structure:
    q0: Encode → RZ → CNOT(control) → RY → Measure ⟨Z₀⟩ → μ₀(x₁,x₂)
                      ↓
    q1: Encode ───── CNOT(target) ────→ Measure ⟨Z₁⟩ → μ₁(x₁,x₂)

    Output: [batch, 2] - Joint membership values from two qubits
    """

    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 2

            # Trainable rotation gates
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)

            # Single CNOT (reduced entanglement)
            self.cnot = tq.CNOT()

        def forward(self, device: tq.QuantumDevice):
            self.rz0(device, wires=0)
            self.cnot(device, wires=[0, 1])  # Only one CNOT operation
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

        measurements = self.measure(device)
        qout = (measurements + 1) / 2
        return qout


# ==================== Quantum Fuzzy Layer: 3x3 Grid Partition Sampling ====================
class JointMembership_FuzzyLayer_JAFFE_NoPool(nn.Module):
    """
    Joint Membership Quantum Fuzzy Layer with 3×3 Grid Partition Sampling (JAFFE-specific)
    """

    def __init__(self, num_classes=7, n_random_pairs=154):
        super(JointMembership_FuzzyLayer_JAFFE_NoPool, self).__init__()
        self.num_classes = num_classes
        self.pooled_size = 32
        self.pixel_dim = 1024
        self.grid_size = 3
        self.block_sizes = [0, 11, 21, 32]

        self.total_pairs = 512
        self.n_random_pairs = n_random_pairs
        self.n_fixed_pairs = self.total_pairs - n_random_pairs

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

    def _compute_block_partition(self):
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

        return all_pairs

    def _sample_fixed_pairs(self, all_pairs, n_fixed):

        if n_fixed <= 0:
            return torch.tensor([], dtype=torch.long).reshape(0, 2)

        n_fixed = min(n_fixed, len(all_pairs))

        rng = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(all_pairs), generator=rng)[:n_fixed]
        sampled_pairs = [all_pairs[i] for i in indices]
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

        return torch.tensor(inter_pairs, dtype=torch.long)

    def _precompute_valid_pixels(self):
        valid_pixels = []
        for block_id in range(9):
            pixels = self.block_pixels[block_id]
            valid = pixels[pixels != -1]
            valid_pixels.append(valid)
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

        return sampling_plan

    def _sample_inter_block_pairs_optimized(self, batch_size, device):
        if self.n_random_pairs == 0:
            return None

        all_pairs = []

        for b1, b2, n_samples in self.sampling_plan:
            pixels_b1 = self.valid_pixels_per_block[b1].to(device)
            pixels_b2 = self.valid_pixels_per_block[b2].to(device)
            n1, n2 = pixels_b1.size(0), pixels_b2.size(0)

            idx1 = torch.randint(0, n1, (batch_size, n_samples), device=device)
            idx2 = torch.randint(0, n2, (batch_size, n_samples), device=device)

            p1 = pixels_b1[idx1]
            p2 = pixels_b2[idx2]

            pairs = torch.stack([p1, p2], dim=-1)
            all_pairs.append(pairs)

        return torch.cat(all_pairs, dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        if self.n_fixed_pairs > 0:
            fixed_pairs_pixels = x_flat[:, self.intra_pair_indices]
        else:
            fixed_pairs_pixels = None

        if self.n_random_pairs > 0:
            random_pairs_pixels = self._sample_inter_block_pairs_optimized(
                batch_size, x.device
            )
        else:
            random_pairs_pixels = None

        if fixed_pairs_pixels is not None and random_pairs_pixels is not None:
            all_pairs_pixels = torch.cat([fixed_pairs_pixels, random_pairs_pixels], dim=1)
        elif fixed_pairs_pixels is not None:
            all_pairs_pixels = fixed_pairs_pixels
        elif random_pairs_pixels is not None:
            all_pairs_pixels = random_pairs_pixels
        else:

            raise ValueError("At least one pairing strategy is required (fixed or random)")
        outputs = []
        for qfuzzier in self.qfuzziers:
            pairs_flat = all_pairs_pixels.view(-1, 2)
            qout = qfuzzier(pairs_flat)
            qout = qout.view(batch_size, self.total_pairs, 2)
            class_output = qout.view(batch_size, -1)
            outputs.append(class_output)

        return torch.stack(outputs, dim=-1)


# ==================== Classical CNN Part ====================
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
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.res1(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.res2(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ==================== JAFFE Complete Network ====================
class JAFFE_JointMembership(nn.Module):
    """
    JAFFE Dual-Qubit Joint Membership Hybrid Network (Unrestricted Version)

    1. Removed all range restrictions for n_random_pairs
    2. Uses new quantum circuit (single CNOT)
    3. Cleaned up mutual information interfaces

    Parameters:
        num_classes: Number of classes (JAFFE=7)
        hidden_dim: Hidden layer dimension
        n_qnn_layers: Reserved parameter (for compatibility)
        n_random_pairs: Number of random pairs (0-512 allowed)
    """

    def __init__(self, num_classes=7, hidden_dim=128, n_qnn_layers=3, n_random_pairs=154):
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

    def forward(self, x):
        batch_size = x.size(0)

        classical_features = self.classical_cnn(x)

        fuzzy_membership = self.quantum_fuzzy(x)
        fuzzy_membership_clamped = torch.clamp(fuzzy_membership, min=1e-7, max=1.0)
        log_membership = torch.log(fuzzy_membership_clamped)
        fuzzy_features = torch.mean(log_membership, dim=1)
        fuzzy_features = self.fusion_fc(fuzzy_features)

        fused_features = classical_features + fuzzy_features

        output = self.classifier(fused_features)

        return output
