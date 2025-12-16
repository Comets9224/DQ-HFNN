
"""
CIFAR-10 Single-Qubit QAHFNN Architecture
- Classical Branch: Maintains DQHFNN's ResNet-style CNN (completely unchanged)
- Quantum Branch: Single-qubit + 3-layer data-reuploading (migrated from Dirty MNIST)
- Processing Strategy: Processes 3072 pixels individually (no pooling)
- Class Name: Fully compatible with main program naming conventions (plug-and-play)
"""

import torch
import torch.nn as nn
import torchquantum as tq


# ==================== single qubit model（3 layer data-reuploading）====================
class SingleQubit_QuantumCircuit(nn.Module):
    """
    Circuit Structure (Per Layer):
    q0: Ry(x) → Rz(θ₁) → Ry(θ₂) → Rz(θ₃) → Measure ⟨Z⟩

    Key Configurations:
    - Number of Qubits: 1
    - Number of Layers: 3 (paper hyperparameter)
    - Trainable Parameters: 9 (3 rotation gates per layer × 3 layers)
    - Data Encoding: Ry rotation (data reuploaded each layer)
    """

    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 1
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz1 = tq.RZ(has_params=True, trainable=True)

        def forward(self, device: tq.QuantumDevice):
            self.rz0(device, wires=0)
            self.ry0(device, wires=0)
            self.rz1(device, wires=0)

    def __init__(self):
        super().__init__()
        self.n_wires = 1

        # Data encoding layer (Ry rotation)
        self.encoder = tq.GeneralEncoder([
            {'input_idx': [0], 'func': 'ry', 'wires': [0]},
        ])

        # 3 trainable circuit blocks (paper configuration)
        self.q_layer1 = self.QLayer()
        self.q_layer2 = self.QLayer()
        self.q_layer3 = self.QLayer()

        # Pauli-Z  mersurement
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):

        bsz = x.shape[0]
        device = tq.QuantumDevice(n_wires=1, bsz=bsz, device=x.device)


        self.encoder(device, x)
        self.q_layer1(device)


        self.encoder(device, x)
        self.q_layer2(device)


        self.encoder(device, x)
        self.q_layer3(device)


        measurement = self.measure(device)  # [batch*3072, 1]
        qout = (measurement.squeeze() + 1) / 2  # [batch*3072]

        return qout


# ================== Quantum Fuzzy Layer (Single-Qubit Version - CIFAR-10 Adaptation) =========
class SingleQubit_FuzzyLayer(nn.Module):
    """
    CIFAR-10 Single-Qubit Fuzzy Layer

    Key Features:
    1. **No Pooling**: Directly processes 32×32×3=3072 pixels
    2. **Pixel-wise Processing**: Each pixel independently through single-qubit circuit
    3. **10 Quantum Circuits**: One single-qubit circuit per class
    4. **Batch Parallelism**: All pixels processed in one batch

    Input: [batch, 3, 32, 32] - RGB images
    Output: [batch, 3072, num_classes] - Membership degree per pixel per class

    Computational Cost: 3072 × num_classes quantum circuit calls/sample
    """

    def __init__(self, num_classes):
        super(JointMembership_FuzzyLayer, self).__init__()
        self.num_classes = num_classes
        self.pixel_dim = 3072  # 32×32×3
        self.qfuzziers = nn.ModuleList([
            SingleQubit_QuantumCircuit() for _ in range(num_classes)
        ])

        print(f"\n{'=' * 70}")
        print(f"【CIFAR-10 Single-Qubit Fuzzy Layer】")
        print(f"{'=' * 70}")
        print(f"  Quantum Circuit Configuration:")
        print(f"    - Input Size: 32×32×3 = {self.pixel_dim} pixels")
        print(f"    - Pooling Strategy: No pooling (direct processing)")
        print(f"    - Quantum Circuits: {num_classes} single-qubit circuits")
        print(f"    - Circuit Layers: 3-layer data-reuploading")
        print(f"    - Trainable Parameters: 9/circuit × {num_classes} = {9 * num_classes}")
        print(f"\n  Computational Cost:")
        print(f"    - Quantum Circuit Calls/Sample: {self.pixel_dim} × {num_classes} = {self.pixel_dim * num_classes}")
        print(f"    - Output Dimension: [batch, {self.pixel_dim}, {num_classes}]")
        print(f"    - GPU Parallelism: 100% (batch vectorization)")
        print(f"\n  Estimated Performance:")
        print(f"    - Training Time: ~20-30 minutes/epoch (GPU dependent)")
        print(f"    - Compared to Dirty MNIST: Computation×3.9 (3072/784)")
        print(f"{'=' * 70}\n")

    def forward(self, x):

        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # [batch, 3072]
        x_all = x_flat.view(-1, 1)  # [batch*3072, 1]
        outputs = []
        for qfuzzier in self.qfuzziers:
            qout = qfuzzier(x_all)  # [batch*3072]
            qout = qout.view(batch_size, self.pixel_dim)
            outputs.append(qout)
        return torch.stack(outputs, dim=-1)


# =========== Classical CNN Feature Extractor (Identical to DQHFNN) =========
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


# =========== Complete QAHFNN Network (Single-Qubit Version) ===========
class Cifar10_QAHFNN_SingleQubit(nn.Module):
    """
    CIFAR-10 Single-Qubit QAHFNN Architecture

    Architecture Components:
    1. Classical Branch: ResNet-style CNN (identical to DQHFNN)
    2. Quantum Branch: Single-qubit fuzzy layer (no pooling, processes 3072 pixels individually)
    3. Fusion Layer: Classical features + quantum features
    4. Classifier: Fully connected layer

    Quantum Circuit Configuration:
    - **Single-qubit**: One quantum circuit per class
    - **3-layer data-reuploading**: Data re-encoded each layer
    - **No pooling**: Directly processes 3072 pixels
    - **Pixel-wise processing**: Independent quantum computation per pixel

    Key Dimensional Flow:
    - Input: [batch, 3, 32, 32]
    - Flattened: [batch, 3072]
    - Quantum Output: [batch, 3072, 10]
    - Fuzzy Rules: [batch, 10]
    - Fusion: [batch, 256]
    - Classification: [batch, 10]

    Computational Cost:
    - Quantum circuit calls/sample: 3072 × 10 = 30720
    - Estimated training time: ~20-30 minutes/epoch (GPU dependent)

    Fuzzy Logic:
    - Uses torch.prod to implement fuzzy "AND" logic
    - Maintains original implementation from author's paper
    - ⚠️ Note: Multiplying 3072 values may cause numerical underflow and gradient vanishing

    Parameters:
    - num_classes: Number of classes (default=10)
    - hidden_dim: Fusion layer dimension (default=256, consistent with DQHFNN)
    - n_random_pairs: Compatibility parameter (ignored in single-qubit architecture)
    """

    def __init__(self, num_classes=10, hidden_dim=256, n_random_pairs=0):
        super(Cifar10_JointMembership, self).__init__()
        self.k = num_classes
        if n_random_pairs != 0:
            print(f"  ⚠️  warning: n_random_pairs={n_random_pairs} in single model be ignored")


        self.class_layer = classical_layer()


        self.qfuzzy_layer = SingleQubit_FuzzyLayer(num_classes=num_classes)


        self.flinear = nn.Linear(self.k, hidden_dim)
        self.classi = nn.Linear(hidden_dim, self.k)

        print(f"\n{'=' * 70}")
        print(f"【CIFAR-10 Single-Qubit QAHFNN Network Configuration】")
        print(f"{'=' * 70}")
        print(f"  Architecture Configuration:")
        print(f"    - Classical Branch: ResNet-style CNN → {hidden_dim}D")
        print(f"    - Quantum Branch: Single-qubit×3072 → {num_classes}D")
        print(f"    - Fuzzy Rules: torch.prod (original AND logic)")
        print(f"    - Fusion Method: torch.add")
        print(f"    - Classifier: {hidden_dim} → {num_classes}")
        print(f"\n  Dimensional Flow:")
        print(f"    - Input: [batch, 3, 32, 32]")
        print(f"    - CNN Features: [batch, {hidden_dim}]")
        print(f"    - Quantum Output: [batch, 3072, {num_classes}]")
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
        print(f"\n  Computational Cost:")
        print(f"    - Quantum Circuit Calls/Sample: 3072 × {num_classes} = {3072 * num_classes}")
        print(f"    - Compared to Dirty MNIST: Computation×3.9")
        print(f"\n  Comparative Experiments:")
        print(f"    - Baseline Model: DNN (Pure Classical)")
        print(f"    - Classical Fuzzy: FDNN (Gaussian Membership)")
        print(f"    - Dual-Qubit: DQHFNN (Dual-Qubit + Entanglement)")
        print(f"    - Single-Qubit: QAHFNN (This Model)")
        print(f"{'=' * 70}\n")

    def forward(self, x):

        batch_size = x.size(0)


        c_part = self.class_layer(x)  # [batch, 256]


        fuzzy_output = self.qfuzzy_layer(x)  # [batch, 3072, 10]


        fuzzy_rule_output = torch.prod(fuzzy_output, dim=1)  # [batch, 10]


        fusion_output = torch.add(c_part, self.flinear(fuzzy_rule_output))


        output = self.classi(fusion_output)
        return output
