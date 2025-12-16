"""
Scene15 DNN (Pure Classical Model)
- Uses only deep neural network for classification
- No fuzzy logic, no quantum circuits
- Serves as performance baseline
- Strictly reproduces author's original implementation
"""
import torch
import torch.nn as nn
import torch.nn.init as init


# ==================== 经典特征提取器 ====================
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


# ==================== Complete DNN Model ====================
class SCENE15_JointMembership(nn.Module):
    """
    Scene15 Pure Classical DNN Model (Strictly Reproducing Author's Implementation)

    Architecture:
        Feature Extractor (3-layer FC) → Classifier (4-layer FC)

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

        self.feature_extractor = DenseNet(input_dim=200, hidden_dim=hidden_dim)

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
        print(f"【Scene15 DNN Pure Classical Model - Strict Reproduction】")
        print(f"{'=' * 70}")
        print(f"  Architecture: Feature Extractor (3-layer FC) → Classifier (4-layer FC)")
        print(f"  Input: [batch, 200] SIFT features")
        print(f"  Hidden Dimension: {hidden_dim}")
        print(f"  Output: [batch, {num_classes}] class probabilities")
        print(f"\n  Key Implementation Details:")
        print(f"    - Feature Extractor: Dropout(0.4) before linear layers")
        print(f"    - Classifier: 4 linear layers + Dropout(0)")
        print(f"    - Parameter Initialization: Based on class_num*2 = {num_classes * 2}")

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n  Model Scale:")
        print(f"    - Total Parameters: {total_params:,}")
        print(f"{'=' * 70}\n")

    def forward(self, x):

        features = self.feature_extractor(x)  # [batch, hidden_dim]
        output = self.classifier(features)  # [batch, num_classes]

        return output

    def get_fused_features(self, x):

        return self.feature_extractor(x)

    def get_classical_features(self, x):

        return self.feature_extractor(x)

    def get_quantum_features(self, x):

        batch_size = x.size(0)
        return torch.zeros(batch_size, self.hidden_dim, device=x.device)
