__version__ = "1.0.1"
try:
    from mamba_ssm_local.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn
    from mamba_ssm_local.modules.mamba_simple import Mamba
    from mamba_ssm_local.models.mixer_seq_simple import MambaLMHeadModel

except:
    from .ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn
    from .modules.mamba_simple import Mamba
    from .models.mixer_seq_simple import MambaLMHeadModel
