import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):

    """
    Low-Rank Adaptation module applied on top of an existing Linear layer.

    Parameters
    linear_layer : nn.Linear
        Base linear projection.
    r : int
        Rank of LoRA decomposition.
    alpha : int
        Scaling factor.
    dropout : float
        Drop probability on input.
    device : torch.device or None
        Target computation device.

    Key components
    lora_A : low-rank projector.
    lora_B : reconstruction matrix.
    scaling : alpha / r factor.
    dropout : regularization layer.

    Returns
    Tensor
        Output = base_linear(x) + LoRA(x) * scaling.
    """

    def __init__(self, linear_layer, r=8, alpha=16, dropout=0.05, device=None):
        super().__init__()
        self.linear = linear_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.device = device or next(linear_layer.parameters()).device

        self.lora_A = nn.Linear(linear_layer.in_features, r, bias=False).to(self.device)
        self.lora_B = nn.Linear(r, linear_layer.out_features, bias=False).to(self.device)
        self.dropout = nn.Dropout(dropout)

        # Inicialización pequeña
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # aseguro que el input esté en el mismo device que los pesos
        x = x.to(self.device)
        return self.linear(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def apply_lora(model, cfg):
    """
    classification layer with LoRALinear.

     Parameters
     model : nn.Module
         Recurrent classifier (RNN/LSTM/GRU).
     cfg : Config
         Configuration object.

     Key arguments
     LORA_R : rank factor.
     LORA_ALPHA : scaling factor.
     LORA_DROPOUT : LoRA dropout.
     device : compute device.

     Returns
     model : nn.Module
         Model with LoRA-augmented final layer.
     """
    print("🔧 Aplicando LoRA manual a la capa final (con soporte CUDA)...")
    device = cfg.DEVICE
    model.fc = LoRALinear(
        model.fc,
        r=cfg.LORA_R,
        alpha=cfg.LORA_ALPHA,
        dropout=cfg.LORA_DROPOUT,
        device=device
    )
    model.to(device)
    return model
