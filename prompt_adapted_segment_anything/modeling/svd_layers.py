import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

class SVDLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.U, self.S, self.Vt = torch.linalg.svd(self.weight, full_matrices=False)
        self.trainable_scale = nn.Parameter(torch.ones_like(self.S))
        self.trainable_shift = nn.Parameter(torch.zeros_like(self.S))
        self.weight.requires_grad = False
        self.done_svd = False
        self.reset_parameters()

    def perform_svd(self):
        self.U, self.S, self.Vt = torch.linalg.svd(self.weight, full_matrices=False)
        self.done_svd = True

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'trainable_shift'):
            nn.init.zeros_(self.trainable_shift)
        if hasattr(self, 'trainable_scale'):
            nn.init.ones_(self.trainable_scale)

    def forward(self, input: torch.Tensor):
        if not self.done_svd:
            self.perform_svd()
        weight_updated = self.U.to(input.device, dtype=input.dtype) @ torch.diag(F.relu(self.trainable_scale.to(input.device, dtype=input.dtype)*self.S.to(input.device, dtype=input.dtype) + self.trainable_shift)) @ self.Vt.to(device=input.device, dtype=input.dtype)
        return F.linear(input, weight_updated, self.bias)

#adapted from https://github.com/phymhan/SVDiff
class SVDConv2d(nn.Conv2d):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        scale: float = 1.0,
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        assert type(kernel_size) is int
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        self.U, self.S, self.Vt = torch.linalg.svd(weight_reshaped, full_matrices=False)
        # initialize to 0 for smooth tuning 
        self.trainable_scale = nn.Parameter(torch.ones_like(self.S))
        self.trainable_shift = nn.Parameter(torch.zeros_like(self.S))
        self.weight.requires_grad = False
        self.done_svd = False
        self.reset_parameters()

    def perform_svd(self):
        # shape
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        self.U, self.S, self.Vt = torch.linalg.svd(weight_reshaped, full_matrices=False)
        self.done_svd = True        
        
    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'trainable_shift'):
            nn.init.zeros_(self.trainable_shift)
        if hasattr(self, 'trainable_scale'):
            nn.init.ones_(self.trainable_scale)

    def forward(self, x: torch.Tensor):
        if not self.done_svd:
            # this happens after loading the state dict 
            self.perform_svd()
        weight_updated = self.U.to(x.device, dtype=x.dtype) @ torch.diag(F.relu(self.trainable_scale.to(x.device, dtype=x.dtype)*self.S.to(x.device, dtype=x.dtype) + self.trainable_shift)) @ self.Vt.to(device=x.device, dtype=x.dtype) 
        weight_updated = rearrange(weight_updated, 'co (cin h w) -> co cin h w', cin=self.weight.size(1), h=self.weight.size(2), w=self.weight.size(3))
        return F.conv2d(x, weight_updated, self.bias, self.stride, self.padding, self.dilation, self.groups)
