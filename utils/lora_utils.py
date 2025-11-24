import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
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
    Aplica LoRA sobre la capa final del modelo recurrente.
    """
    print("Aplicando LoRA manual a la capa final (con soporte CUDA)...")
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
