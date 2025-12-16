"""
Scene15 QA-HFNN (Quantum-Assisted Hierarchical Fuzzy Neural Network) - Three-Layer Quantum Circuit Version
- Classical DNN + Quantum Fuzzy Logic
- Quantum circuit upgraded from single-layer to three-layer (referencing JAFFE implementation)
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torchquantum as tq


# ==================== Classical Feature Extractor ====================
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


# ==================== Quantum Fuzzy Layer (Three-Layer Version) ====================
class Q_Model_3Layer(nn.Module):
    """
    Single Quantum Membership Function (Three-Layer Version)
    Uses single-qubit + data re-uploading

    References SingleQubitQNN structure from JAFFE implementation
    """

    def __init__(self, n_layers=3):
        super(Q_Model_3Layer, self).__init__()
        self.n_wires = 1
        self.n_layers = n_layers

        self.encoder = tq.GeneralEncoder([
            {'input_idx': [0], 'func': 'ry', 'wires': [0]},
        ])

        self.q_layers = nn.ModuleList([
            self._build_layer() for _ in range(n_layers)
        ])

        self.measure = tq.MeasureAll(tq.PauliZ)

    def _build_layer(self):
        return nn.ModuleList([
            tq.RZ(has_params=True, trainable=True),
            tq.RY(has_params=True, trainable=True),
            tq.RZ(has_params=True, trainable=True)
        ])

    def forward(self, x):
        bsz = x.shape[0]
        x = x.view(-1, 1)

        device = tq.QuantumDevice(n_wires=1, bsz=x.shape[0], device=x.device)

        for layer in self.q_layers:
            self.encoder(device, x)

            layer[0](device, wires=0)
            layer[1](device, wires=0)
            layer[2](device, wires=0)

        x = self.measure(device)

        qout = (x + 1) / 2

        return qout


class QFuzzyLayer_3Layer(nn.Module):
    def __init__(self, k, n_layers=3):
        super(QFuzzyLayer_3Layer, self).__init__()
        self.num_memberships = k
        self.n_layers = n_layers

        self.qfuzziers = nn.ModuleList([
            Q_Model_3Layer(n_layers=n_layers) for _ in range(k)
        ])

    def forward(self, x):
        outputs = []
        for qfuzzier in self.qfuzziers:
            outputs.append(qfuzzier(x))
        return torch.stack(outputs, dim=-1)


# ==================== Complete QA-HFNN Model ====================
class SCENE15_JointMembership(nn.Module):
    """
    Scene15 Quantum-Assisted Hierarchical Fuzzy Neural Network (QA-HFNN) - Three-Layer Quantum Circuit Version

    Architecture:
        1. Classical Branch: 3-layer FC feature extractor
        2. Quantum Fuzzy Branch: Single-qubit membership function (3-layer circuit)
        3. Fusion Layer: ADD fusion
        4. Classifier: 4-layer FC

    Parameters:
        num_classes: Number of classes (Scene15=15)
        hidden_dim: Hidden dimension (default 256)
        n_qnn_layers: Quantum circuit layers (now defaults to 3)
        n_random_pairs: Reserved parameter (interface compatibility, unused)
    """

    def __init__(self, num_classes=15, hidden_dim=256, n_qnn_layers=3, n_random_pairs=0):
        super(SCENE15_JointMembership, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.k = num_classes
        self.n_qnn_layers = n_qnn_layers

        self.class_layer = DenseNet(input_dim=200, hidden_dim=hidden_dim)

        self.qfuzzy_layer = QFuzzyLayer_3Layer(num_classes, n_layers=n_qnn_layers)

        self.flinear = nn.Linear(self.k, hidden_dim)

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

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                init.uniform_(m.weight,
                              a=-1 / torch.sqrt(torch.tensor(num_classes * 2, dtype=torch.float32)),
                              b=1 / torch.sqrt(torch.tensor(num_classes * 2, dtype=torch.float32)))
                if m.bias is not None:
                    init.zeros_(m.bias)

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

        print(f"\n{'=' * 70}")
        print(
            f"【Scene15 QA-HFNN Quantum-Assisted Hierarchical Fuzzy Neural Network - Three-Layer Quantum Circuit Version】")
        print(f"{'=' * 70}")
        print(f"  Architecture Components:")
        print(f"    - Classical Branch: 3-layer FC feature extractor")
        print(f"    - Quantum Fuzzy Branch: Single-qubit membership function (3-layer circuit)")
        print(f"    - Fusion Strategy: ADD fusion")
        print(f"\n  Input Processing:")
        print(f"    - Input Size: [batch, 200] SIFT features")
        print(f"    - Hidden Dimension: {hidden_dim}")
        print(f"\n  Quantum Configuration:")
        print(f"    - Quantum Circuits: {num_classes} (one per class)")
        print(f"    - Qubits: 1")
        print(f"    - Circuit Layers: {n_qnn_layers} ← Upgraded to three layers")
        print(f"    - Data Encoding: Angle encoding (Ry)")
        print(f"    - Per-Layer Structure: Ry(x) → Rz(θ1) → Ry(θ2) → Rz(θ3)")
        print(f"    - Total Quantum Parameters: {n_qnn_layers * 3}/circuit")
        print(f"\n  Key Implementation Details:")
        print(f"    - Feature Extractor: Dropout(0.4) before linear layers")
        print(f"    - Classifier: 4 linear layers + Dropout(0)")
        print(f"    - Parameter Initialization: Based on class_num*2 = {num_classes * 2}")
        print(f"\n  Dimension Flow:")
        print(f"    - Quantum Output: [batch, 200, {num_classes}]")
        print(f"    - Fuzzy Rules: [batch, {num_classes}]")
        print(f"    - Fusion Dimension: {hidden_dim}")

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        classical_params = sum(p.numel() for p in self.class_layer.parameters() if p.requires_grad)
        quantum_params = sum(p.numel() for p in self.qfuzzy_layer.parameters() if p.requires_grad)
        print(f"\n  Model Scale:")
        print(f"    - Total Parameters: {total_params:,}")
        print(f"    - Classical Branch: {classical_params:,} ({classical_params / total_params * 100:.2f}%)")
        print(f"    - Quantum Branch: {quantum_params:,} ({quantum_params / total_params * 100:.2f}%)")
        print(f"    - Per QNN: {quantum_params // num_classes} parameters (3 layers × 3 gates)")
        print(f"{'=' * 70}\n")

    def forward(self, x):

        batch_size = x.size(0)
        c_part = self.class_layer(x)  # [batch, hidden_dim]
        x_flat = x.view(-1)  # [batch*200]
        x = self.qfuzzy_layer(x_flat)  # [batch, 200, num_classes]
        x = x.view(batch_size, -1, self.k)  # [batch, 200, k]
        fuzzy_rule_output = torch.prod(x, dim=1)  # [batch, k]
        fuzzied_x = fuzzy_rule_output.view(batch_size, self.k)  # [batch, k]
        fusion_output = torch.add(c_part, self.flinear(fuzzied_x))  # [batch, hidden_dim]
        x = self.classifier(fusion_output)  # [batch, num_classes]

        return x

    def get_fused_features(self, x):

        batch_size = x.size(0)

        c_part = self.class_layer(x)

        x_flat = x.view(-1)
        fuzzy_memberships = self.qfuzzy_layer(x_flat)
        fuzzy_memberships = fuzzy_memberships.view(batch_size, -1, self.num_classes)
        fuzzy_features = torch.prod(fuzzy_memberships, dim=1)
        fuzzy_features = self.flinear(fuzzy_features)

        fused_features = torch.add(c_part, fuzzy_features)

        return fused_features

    def get_classical_features(self, x):

        return self.class_layer(x)

    def get_quantum_features(self, x):

        batch_size = x.size(0)

        x_flat = x.view(-1)
        fuzzy_memberships = self.qfuzzy_layer(x_flat)
        fuzzy_memberships = fuzzy_memberships.view(batch_size, -1, self.num_classes)
        fuzzy_features = torch.prod(fuzzy_memberships, dim=1)
        fuzzy_features = self.flinear(fuzzy_features)

        return fuzzy_features
