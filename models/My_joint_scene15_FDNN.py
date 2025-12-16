"""
Scene15 FDNN (Classical Fuzzy Deep Network)
- Classical DNN + Classical Fuzzy Logic (Gaussian Membership Functions)
- Serves as classical fuzzy baseline for quantum method comparison
- Strictly reproduces author's original implementation
"""

import torch
import torch.nn as nn
import torch.nn.init as init


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


# ==================== Classical Fuzzy Layer ====================
class GaussianMembershipFunction(nn.Module):
    def __init__(self, in_features):
        super(GaussianMembershipFunction, self).__init__()
        self.in_features = in_features

        self.mean = nn.Parameter(torch.randn(in_features))
        self.std = nn.Parameter(torch.ones(in_features) * 0.5)

    def forward(self, x):
        membership = torch.exp(-((x - self.mean) ** 2) / (2 * self.std ** 2 + 1e-8))
        return membership


class FuzzyMembershipLayer(nn.Module):

    def __init__(self, in_features, num_memberships):
        super(FuzzyMembershipLayer, self).__init__()
        self.in_features = in_features
        self.num_memberships = num_memberships

        self.memberships = nn.ModuleList([
            GaussianMembershipFunction(in_features)
            for _ in range(num_memberships)
        ])

    def forward(self, x):
        outputs = []
        for membership_func in self.memberships:
            outputs.append(membership_func(x))
        return torch.stack(outputs, dim=-1)  # [batch, in_features, k]


# ==================== Complete FDNN Model ====================
class SCENE15_JointMembership(nn.Module):
    """
    Scene15 Classical Fuzzy Deep Network (FDNN) - Strictly Reproducing Author's Implementation

    Architecture:
        1. Classical Branch: 3-layer FC feature extractor
        2. Classical Fuzzy Branch: Gaussian membership functions
        3. Fusion Layer: ADD fusion
        4. Classifier: 4-layer FC

    Key Details:
    - Classifier has 4 linear layers (not 3)
    - Classifier Dropout probability is 0 (effectively unused)
    - Parameter initialization based on class_num * 2

    Parameters:
        num_classes: Number of classes (Scene15=15)
        hidden_dim: Hidden dimension (default 256)
    """

    def __init__(self, num_classes=15, hidden_dim=256, n_qnn_layers=1, n_random_pairs=0):
        super(SCENE15_JointMembership, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.classical_branch = DenseNet(input_dim=200, hidden_dim=hidden_dim)

        self.fuzzy_layer = FuzzyMembershipLayer(
            in_features=200,
            num_memberships=num_classes
        )

        self.fusion_fc = nn.Linear(num_classes, hidden_dim)

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

        print(f"\n{'=' * 70}")
        print(f"【Scene15 FDNN Classical Fuzzy Deep Network - Strict Reproduction】")
        print(f"{'=' * 70}")
        print(f"  Architecture Components:")
        print(f"    - Classical Branch: 3-layer FC feature extractor")
        print(f"    - Classical Fuzzy Branch: Gaussian membership functions")
        print(f"    - Fusion Strategy: ADD fusion")
        print(f"\n  Input Processing:")
        print(f"    - Input Size: [batch, 200] SIFT features")
        print(f"    - Hidden Dimension: {hidden_dim}")
        print(f"\n  Fuzzy Configuration:")
        print(f"    - Membership Function: Gaussian")
        print(f"    - Function Count: {num_classes} (one per class)")
        print(f"    - Trainable Parameters: Mean + Standard Deviation")
        print(f"\n  Key Implementation Details:")
        print(f"    - Feature Extractor: Dropout(0.4) before linear layers")
        print(f"    - Classifier: 4 linear layers + Dropout(0)")
        print(f"    - Parameter Initialization: Based on class_num*2 = {num_classes * 2}")
        print(f"\n  Dimension Flow:")
        print(f"    - Fuzzy Output: [batch, 200, {num_classes}]")
        print(f"    - Fuzzy Rules: [batch, {num_classes}]")
        print(f"    - Fusion Dimension: {hidden_dim}")

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        classical_params = sum(p.numel() for p in self.classical_branch.parameters() if p.requires_grad)
        fuzzy_params = sum(p.numel() for p in self.fuzzy_layer.parameters() if p.requires_grad)
        print(f"\n  Model Scale:")
        print(f"    - Total Parameters: {total_params:,}")
        print(f"    - Classical Branch: {classical_params:,} ({classical_params / total_params * 100:.2f}%)")
        print(f"    - Fuzzy Branch: {fuzzy_params:,} ({fuzzy_params / total_params * 100:.2f}%)")
        print(f"{'=' * 70}\n")

    def forward(self, x):

        batch_size = x.size(0)

        classical_features = self.classical_branch(x)  # [batch, hidden_dim]

        fuzzy_memberships = self.fuzzy_layer(x)  # [batch, 200, num_classes]

        fuzzy_features = torch.prod(fuzzy_memberships, dim=1)  # [batch, num_classes]

        fuzzy_features = self.fusion_fc(fuzzy_features)  # [batch, hidden_dim]

        fused_features = classical_features + fuzzy_features  # [batch, hidden_dim]

        output = self.classifier(fused_features)  # [batch, num_classes]

        return output

    def get_fused_features(self, x):

        batch_size = x.size(0)

        classical_features = self.classical_branch(x)

        fuzzy_memberships = self.fuzzy_layer(x)
        fuzzy_features = torch.prod(fuzzy_memberships, dim=1)
        fuzzy_features = self.fusion_fc(fuzzy_features)

        fused_features = classical_features + fuzzy_features

        return fused_features

    def get_classical_features(self, x):

        return self.classical_branch(x)

    def get_quantum_features(self, x):

        fuzzy_memberships = self.fuzzy_layer(x)
        fuzzy_features = torch.prod(fuzzy_memberships, dim=1)
        fuzzy_features = self.fusion_fc(fuzzy_features)
        return fuzzy_features
