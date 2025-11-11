from typing import Dict, Optional
import torch
from torch import nn
import importlib, sys, os
from collections import OrderedDict


from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict

@dataclass(frozen=True)
class VQEncoderCfg:
    ckpt: str
    repo_path: Optional[str] = None
    image_size: int = 256
    encoder_import: str = "taming.modules.diffusionmodules.improved_model.Encoder"
    state_prefix: str = "encoder."
    encoder_kwargs: Dict = field(default_factory=lambda: dict(
        in_channels=3, out_channels=256, ch=128,
        ch_mult=[1,1,2,2,4], num_res_blocks=2,
        double_z=False, attn_resolutions=[]
    ))


def ensure_on_sys_path(path: str):
    path = os.path.abspath(path)
    if path not in sys.path:
        sys.path.insert(0, path)

def lazy_import(import_path: str, repo_path: str = None):
    if repo_path:
        ensure_on_sys_path(repo_path)  
    module_path, attr = import_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)

def filter_state_dict(sd, prefix):
    return OrderedDict((k[len(prefix):], v) for k, v in sd.items() if k.startswith(prefix))

class VQEncoder(nn.Module):
    def __init__(self, cfg: VQEncoderCfg):
        super().__init__()
        Encoder = lazy_import(cfg.encoder_import, cfg.repo_path)  
        self.encoder = Encoder(**cfg.encoder_kwargs).eval()
        sd = filter_state_dict(torch.load(cfg.ckpt, map_location="cpu")["state_dict"], cfg.state_prefix)
        self.encoder.load_state_dict(sd, strict=False)

    @torch.no_grad()
    def encode(self, x):
        return self.encoder(x)
    
    @torch.no_grad()
    def test_encoder(self, B: int = 2, H: int = 256, W: int = 256,
                     dtype=torch.float32, device: str = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = self.encoder.to(device).eval()

        x = torch.randn(B, 3, H, W, dtype=dtype, device=device)
        y = self.encoder(x)

        print(f"[VQEncoder.test_encoder] input  : {tuple(x.shape)}")
        print(f"[VQEncoder.test_encoder] output : {tuple(y.shape)}")

    
    
    
if __name__ == "__main__":
    
    cfg_dict = {
        "ckpt": "/home/110/u110003/ckpt/vqgan/vqgan_imagenet_f16_1024/checkpoints/last.ckpt",
        "repo_path": "/home/110/u110003/code/1111/vqcode/SEED-Voken-main",
        "encoder_kwargs": {"double_z": False,
                            "z_channels": 256,
                            "resolution": 256,
                            "in_channels": 3,
                            "out_ch": 3,
                            "ch": 128,
                            "ch_mult": [ 1,1,2,2,4],
                            "num_res_blocks": 2,
                            "attn_resolutions": [16],
                            "dropout": 0.0},
        "encoder_import": "src.IBQ.modules.diffusionmodules.model.Encoder"
        }

    cfg = VQEncoderCfg(**cfg_dict)
    enc = VQEncoder(cfg)
    enc.test_encoder()


'''
source /home/110/u110003/uv_env/seed/.venv/bin/activate
cd /home/110/u110003/code/1111
srun --gpus=1 --partition=debug-A10-01 --ntasks-per-node=1 python encoder_factory.py
'''