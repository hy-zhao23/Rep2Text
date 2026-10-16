import torch.nn as nn
import torch
import math
from typing import Tuple
from utils import p_args, logger, gh

class EmbeddingProjector(nn.Module):
    '''
    Map an input embedding to N vectors
    '''
    def __init__(self, 
        dim_in: int, 
        dim_out: int,
        eps = 1e-5, 
        bias: bool = False, # LayerNorm makes bias unnecessary
        res_gain: float = 1.0,     # small residual gate
    ):
        super().__init__()
        self.n_vecs = p_args.proj.n_vecs
        self.drop_rate = p_args.proj.ei_proj_dropout

        assert self.n_vecs > 0, "n_vecs must be greater than 0"

        self.dim_in = dim_in
        self.dim_out = dim_out
        _dtype = torch.bfloat16
        hidden_dim = int(dim_in * p_args.proj.ei_proj_expansion)  # Expand then contract
        self.res_gain = res_gain

        # Pre/post norms - create in bfloat16 to match saved parameters
        self.in_norm  = nn.LayerNorm(dim_in,  eps=eps, dtype=_dtype)
        self.out_norm = nn.LayerNorm(dim_out, eps=eps, dtype=_dtype)
        
        self.fc1 = nn.Linear(dim_in, hidden_dim, bias=bias)
        self.act = nn.GELU()
        self.drop = nn.Dropout(self.drop_rate) if self.drop_rate and self.drop_rate > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, dim_out * self.n_vecs, bias=bias)

        # Residual path: project input to dim_out, then broadcast to N
        self.skip = nn.Identity() if dim_in == dim_out else nn.Linear(dim_in, dim_out, bias=False)
        self.gate = nn.Parameter(torch.full((self.n_vecs,), self.res_gain, dtype=_dtype))
        self._print_params()

    @torch.no_grad()
    def reset_parameters(self):

        # fc1: Kaiming for GELU (fan_in)
        nn.init.kaiming_uniform_(self.fc1.weight, a=0.0, mode="fan_in", nonlinearity="relu")
        if self.fc1.bias is not None and self.fc1.bias.numel() > 0:
            nn.init.zeros_(self.fc1.bias)

        # fc2: small init to prevent early overconfident logits
        if self.fc2.weight.dim() >= 2:  # Check if tensor has at least 2 dimensions
            fan_in = self.fc2.weight.size(1)
            std = math.sqrt(2.0 / fan_in) * 0.5
            self.fc2.weight.normal_(0.0, std)
        if self.fc2.bias is not None and self.fc2.bias.numel() > 0:
            nn.init.zeros_(self.fc2.bias)

        # Skip: identity-like top-left block if expanding; Xavier small if contracting
        if isinstance(self.skip, nn.Linear):
            w = self.skip.weight  # [out, in]
            nn.init.zeros_(w)
            out, in_ = w.shape
            k = min(out, in_)
            # fill top-left kxk with identity
            w[:k, :k].fill_diagonal_(1.0)
            if out < in_:  # contracting: make it a bit smaller overall
                nn.init.xavier_uniform_(w, gain=0.1)

        # Norms
        self.in_norm.weight.data.fill_(1.0)
        self.in_norm.bias.data.zero_()
        self.out_norm.weight.data.fill_(1.0)
        self.out_norm.bias.data.zero_()

        # Residual gate
        self.gate.data.clamp_(0.0, 1.0)
        self.gate.data.fill_(self.res_gain)
        logger.info("🔍 Projector parameters reset")

    def forward(self, x):
        # Use bfloat16 consistently - no dtype conversions needed
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        
        orig_shape = x.shape
        assert x.size(-1) == self.dim_in, f"Expected last dim {self.dim_in}, got {x.size(-1)}"

        # Flatten leading dims except feature dim
        leading_shape: Tuple[int, ...] = orig_shape[:-1]

        res = self.skip(x)

        h = self.fc1(self.in_norm(x))
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        
        h = h.view(-1, self.n_vecs, self.dim_out)

        # Apply per-output gate: gate shape [N] -> (1, N, 1)
        gated = res.unsqueeze(1) + self.gate.view(1, self.n_vecs, 1) * h  # (L, N, dim_out)

        # LayerNorm over the last dim independently for each output
        # Flatten (L*N, dim_out) -> norm -> reshape back
        LN_in = gated.reshape(-1, self.dim_out)
        LN_out = self.out_norm(LN_in)
        y = LN_out.reshape(-1, self.n_vecs, self.dim_out)  # (L, N, dim_out)

        # Restore batch/sequence dims
        y = y.reshape(*leading_shape, self.n_vecs, self.dim_out)   # (…, N, dim_out)

        return y
        
    def _print_params(self):
        if gh.is_main:
            logger.info("-------------------------------------------------------------------")
            logger.info("--------------------Projector Parameter Settings-------------------")
            logger.info("-------------------------------------------------------------------")
            logger.info(f"EmbeddingProjector: {self.dim_in} -> {self.dim_out}")
            logger.info(f"Expansion: {p_args.proj.ei_proj_expansion}")
            logger.info(f"Dropout: {p_args.proj.ei_proj_dropout}")
            logger.info(f"Residual Gain Init: {self.res_gain}")
            logger.info(f"Number of projected vectors: {self.n_vecs}")