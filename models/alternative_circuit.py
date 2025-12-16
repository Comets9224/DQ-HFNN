"""
[Architecture Details]
A: Asym-2P-0C-NoEnt      - Rz|0⟩ → Ry|0⟩
B: Asym-2P-2C-StrongEnt  - Rz|0⟩ → CNOT → Ry|0⟩ → CNOT
C: Asym-2P-1C-WeakEnt    - Rz|0⟩ → CNOT → Ry|0⟩
D: Sym-8P-1C-StrongEnt   - [Rx Ry]×2 → CNOT → [Rx Ry]×2
E: Sym-4P-0C-NoEnt-2x2   - Rz|0⟩ Ry|0⟩ Rz|1⟩ Ry|1⟩ (2+2)
F: Sym-12P-2C-DoubleEnt  - [Rx Ry]×2 → CNOT → [Rx Ry]×2 → CNOT → [Rx Ry]×2
G: Asym-4P-0C-NoEnt-3x1  - Rz|0⟩ → Ry|0⟩ → Rz|0⟩ + Ry|1⟩ (3+1)
"""
#Usage:
# Please place the following circuit blocks sequentially into the network layer code, replacing the circuits in the code

# circuit A
class QLayer(tq.QuantumModule):

    def __init__(self):
        super().__init__()
        self.n_wires = 2

        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)

    def forward(self, device: tq.QuantumDevice):
        self.rz0(device, wires=0)
        self.ry0(device, wires=0)


# circuit B
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


# circuit C
class QLayer(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 2

        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)

        self.cnot = tq.CNOT()

    def forward(self, device: tq.QuantumDevice):
        self.cnot(device, wires=[0, 1])
        self.ry0(device, wires=0)


# circuit D
class QLayer(tq.QuantumModule):

    def __init__(self):
        super().__init__()
        self.n_wires = 2

        self.rx0_layer1 = tq.RX(has_params=True, trainable=True)
        self.ry0_layer1 = tq.RY(has_params=True, trainable=True)
        self.rx1_layer1 = tq.RX(has_params=True, trainable=True)
        self.ry1_layer1 = tq.RY(has_params=True, trainable=True)

        self.cnot = tq.CNOT()

        self.rx0_layer2 = tq.RX(has_params=True, trainable=True)
        self.ry0_layer2 = tq.RY(has_params=True, trainable=True)
        self.rx1_layer2 = tq.RX(has_params=True, trainable=True)
        self.ry1_layer2 = tq.RY(has_params=True, trainable=True)

    def forward(self, device: tq.QuantumDevice):
        self.rx0_layer1(device, wires=0)
        self.ry0_layer1(device, wires=0)
        self.rx1_layer1(device, wires=1)
        self.ry1_layer1(device, wires=1)

        self.cnot(device, wires=[0, 1])

        self.rx0_layer2(device, wires=0)
        self.ry0_layer2(device, wires=0)
        self.rx1_layer2(device, wires=1)
        self.ry1_layer2(device, wires=1)


# circuit E
class QLayer(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 2

        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)

        self.rz1 = tq.RZ(has_params=True, trainable=True)
        self.ry1 = tq.RY(has_params=True, trainable=True)

    def forward(self, device: tq.QuantumDevice):
        self.rz0(device, wires=0)
        self.ry0(device, wires=0)

        self.rz1(device, wires=1)
        self.ry1(device, wires=1)


# circuit F
class QLayer(tq.QuantumModule):

    def __init__(self):
        super().__init__()
        self.n_wires = 2

        self.rx0_layer1 = tq.RX(has_params=True, trainable=True)
        self.ry0_layer1 = tq.RY(has_params=True, trainable=True)
        self.rx1_layer1 = tq.RX(has_params=True, trainable=True)
        self.ry1_layer1 = tq.RY(has_params=True, trainable=True)

        self.cnot1 = tq.CNOT()

        self.rx0_layer2 = tq.RX(has_params=True, trainable=True)
        self.ry0_layer2 = tq.RY(has_params=True, trainable=True)
        self.rx1_layer2 = tq.RX(has_params=True, trainable=True)
        self.ry1_layer2 = tq.RY(has_params=True, trainable=True)

        self.cnot2 = tq.CNOT()

        self.rx0_layer3 = tq.RX(has_params=True, trainable=True)
        self.ry0_layer3 = tq.RY(has_params=True, trainable=True)
        self.rx1_layer3 = tq.RX(has_params=True, trainable=True)
        self.ry1_layer3 = tq.RY(has_params=True, trainable=True)

    def forward(self, device: tq.QuantumDevice):
        self.rx0_layer1(device, wires=0)
        self.ry0_layer1(device, wires=0)
        self.rx1_layer1(device, wires=1)
        self.ry1_layer1(device, wires=1)

        self.cnot1(device, wires=[0, 1])

        self.rx0_layer2(device, wires=0)
        self.ry0_layer2(device, wires=0)
        self.rx1_layer2(device, wires=1)
        self.ry1_layer2(device, wires=1)

        self.cnot2(device, wires=[0, 1])

        self.rx0_layer3(device, wires=0)
        self.ry0_layer3(device, wires=0)
        self.rx1_layer3(device, wires=1)
        self.ry1_layer3(device, wires=1)


# circuit G
class QLayer(tq.QuantumModule):

    def __init__(self):
        super().__init__()
        self.n_wires = 2

        self.rz0_a = tq.RZ(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0_b = tq.RZ(has_params=True, trainable=True)

        self.ry1 = tq.RY(has_params=True, trainable=True)

    def forward(self, device: tq.QuantumDevice):
        self.rz0_a(device, wires=0)
        self.ry0(device, wires=0)
        self.rz0_b(device, wires=0)

        self.ry1(device, wires=1)
# Error using
class IncorrectQuantumLayer(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 2
        # Did not instantiate CNOT in __init__

    def forward(self, device: tq.QuantumDevice):
        # Directly calls tq.CNOT(...), which is a temporary and invalid operation
        tq.CNOT(device, wires=[0, 1]) # <--- Error!