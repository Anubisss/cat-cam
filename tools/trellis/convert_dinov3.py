"""Converts timm's ungated copy of DINOv3 ViT-L/16 to the transformers layout
TRELLIS.2 loads, so there's no wait for access to Meta's gated repo (the
weights and their licence are the same). Checks both give the same features.

    .venv/bin/python convert_dinov3.py dinov3-vitl16
"""

import sys

import timm
import torch
import torch.nn.functional as F
from transformers import DINOv3ViTConfig, DINOv3ViTModel

OUT = sys.argv[1]

src = timm.create_model("vit_large_patch16_dinov3.lvd1689m", pretrained=True).eval()
sd = src.state_dict()
cfg = DINOv3ViTConfig(hidden_size=1024, intermediate_size=4096, num_attention_heads=16, num_hidden_layers=24,
                      num_register_tokens=4, query_bias=False, key_bias=False, value_bias=False,
                      layerscale_value=1e-5, image_size=224)
dst = DINOv3ViTModel(cfg).eval()

new = {
    "embeddings.cls_token": sd["cls_token"],
    "embeddings.mask_token": torch.zeros(1, 1, 1024),
    "embeddings.register_tokens": sd["reg_token"],
    "embeddings.patch_embeddings.weight": sd["patch_embed.proj.weight"],
    "embeddings.patch_embeddings.bias": sd["patch_embed.proj.bias"],
    "norm.weight": sd["norm.weight"],
    "norm.bias": sd["norm.bias"],
}
for i in range(24):
    a, b = f"blocks.{i}.", f"model.layer.{i}."
    q, k, v = sd[a + "attn.qkv.weight"].chunk(3)  # timm keeps them in one matrix
    new |= {
        b + "norm1.weight": sd[a + "norm1.weight"], b + "norm1.bias": sd[a + "norm1.bias"],
        b + "attention.q_proj.weight": q, b + "attention.k_proj.weight": k, b + "attention.v_proj.weight": v,
        b + "attention.o_proj.weight": sd[a + "attn.proj.weight"],
        b + "attention.o_proj.bias": sd[a + "attn.proj.bias"],
        b + "layer_scale1.lambda1": sd[a + "gamma_1"], b + "layer_scale2.lambda1": sd[a + "gamma_2"],
        b + "norm2.weight": sd[a + "norm2.weight"], b + "norm2.bias": sd[a + "norm2.bias"],
        b + "mlp.up_proj.weight": sd[a + "mlp.fc1.weight"], b + "mlp.up_proj.bias": sd[a + "mlp.fc1.bias"],
        b + "mlp.down_proj.weight": sd[a + "mlp.fc2.weight"],
        b + "mlp.down_proj.bias": sd[a + "mlp.fc2.bias"],
    }
dst.load_state_dict(new, strict=True)

# The input TRELLIS gives it: 512x512, ImageNet-normalised.
torch.manual_seed(0)
image = torch.rand(1, 3, 512, 512)
mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
image = (image - mean[:, None, None]) / std[:, None, None]

with torch.no_grad():
    # As TRELLIS's DinoV3FeatureExtractor.extract_features does.
    h = dst.embeddings(image, bool_masked_pos=None)
    rope = dst.rope_embeddings(image)
    for layer in dst.model.layer:
        h = layer(h, position_embeddings=rope)
    ours = F.layer_norm(h, h.shape[-1:])

    feats = {}
    src.blocks[-1].register_forward_hook(lambda m, i, o: feats.__setitem__("x", o))
    src.forward_features(image)
    ref = F.layer_norm(feats["x"], feats["x"].shape[-1:])

cos = F.cosine_similarity(ours, ref, dim=-1)
print(f"features agree: min cosine {cos.min():.6f}")
assert cos.min() > 0.9999, "converted model disagrees with timm"
dst.save_pretrained(OUT)
print("saved to", OUT)
