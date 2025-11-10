import math
import os
from typing import Dict
import argparse
import random
from collections import Counter
import glob
import random, shutil
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
from einops import rearrange
from PIL import Image, ImageDraw


from taming.modules.diffusionmodules.improved_model import Encoder, Decoder
from taming.modules.diffusionmodules.cyclemlp import CycleNet, CycleMLP, default_cfgs
from taming.modules.diffusionmodules.convmlp import convmlp_l_encoder
from taming.modules.vqvae.quantize import SimVQ as VectorQuantizer


def draw_token_boxes_on_image(image_path, token_indices, target_token, h_feat, w_feat, save_dir):
    """
    在原图上标出特定 token 的 patch 区域
    Args:
        image_path: 原图路径
        token_indices: tensor, shape=(H_feat*W_feat,)
        target_token: int, 要标记的 token id
        h_feat, w_feat: encoder 输出的 feature map 高宽
        save_dir: 保存路径目录
    """
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    patch_w, patch_h = img_w // w_feat, img_h // h_feat

    token_map = token_indices.view(h_feat, w_feat)
    draw = ImageDraw.Draw(img)
    color = (255, 0, 0)  # 红色框

    count = 0
    for i in range(h_feat):
        for j in range(w_feat):
            if token_map[i, j].item() == target_token:
                x1, y1 = j * patch_w, i * patch_h
                x2, y2 = x1 + patch_w, y1 + patch_h
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                count += 1

    if count > 0:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"token_{target_token}_{os.path.basename(image_path)}")
        img.save(save_path)
        print(f"✅ {os.path.basename(image_path)}: 标注 {count} 个 patch → {save_path}")
    else:
        print(f"⚠️ {os.path.basename(image_path)}: 无 token {target_token} 匹配区域")
        

def get_random_imagenet_images(root_dir, num_classes=5, per_class=1, save_dir="./selected_imagenet_samples"):
    """
    从 ImageNet 数据集中随机选取若干图片，并复制到指定目录。
    
    Args:
        root_dir (str): ImageNet 目录，例如 '/home/xxx/imagenet/train'
        num_classes (int): 选取多少个类别
        per_class (int): 每个类别选取多少张
        save_dir (str): 输出目录，用于保存选取的图片副本
    Returns:
        image_paths (list[str]): 被选取图片的完整路径
    """
    os.makedirs(save_dir, exist_ok=True)

    class_dirs = [d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)]
    selected_classes = random.sample(class_dirs, num_classes)

    image_paths = []
    print(f"🎯 从 {len(class_dirs)} 个类别中随机选取 {num_classes} 类，每类 {per_class} 张：")

    for cls in selected_classes:
        cls_name = os.path.basename(cls)
        imgs = glob.glob(os.path.join(cls, "*.JPEG")) + glob.glob(os.path.join(cls, "*.jpg"))
        if len(imgs) == 0:
            continue
        sampled_imgs = random.sample(imgs, min(per_class, len(imgs)))
        for img in sampled_imgs:
            dst_name = f"{cls_name}_{os.path.basename(img)}"
            dst_path = os.path.join(save_dir, dst_name)
            shutil.copy(img, dst_path)
            image_paths.append(dst_path)

    print(f"✅ 已保存 {len(image_paths)} 张图片到目录：{save_dir}")
    for img in image_paths:
        print(" -", img)

    return image_paths


def visualize_token_rows(row_grids, image_paths, token_id, save_path, per_row_height=64, per_row_limit=5):
    """
    row_grids: list[Tensor(C,H,W)] 或 None     # 每行包含若干 decoded patches；若该图没有token则为 None
    image_paths: list[str]                     # 对应原图路径
    token_id: int                              # 当前 token id
    save_path: str                             # 输出路径
    per_row_height: int                        # 缩略图高度
    per_row_limit: int                         # 每行最多展示多少 patch，不足补灰
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # --- 生成缩略图 ---
    thumbs = []
    for path in image_paths:
        img = Image.open(path).convert("RGB").resize((per_row_height, per_row_height))
        thumbs.append(TF.to_tensor(img))

    # --- 找出所有非 None 行的参考尺寸 ---
    valid_rows = [r for r in row_grids if r is not None]
    if len(valid_rows) > 0:
        c, ref_h, ref_w = valid_rows[0].shape
    else:
        # 全 None，用默认大小
        ref_h, ref_w = per_row_height, per_row_height * per_row_limit

    padded_rows = []
    for i, maybe_row in enumerate(row_grids):
        # 若该图没有 token，整行灰块
        if maybe_row is None:
            gray_patch = torch.zeros((3, ref_h, ref_h)) 
            patches = [gray_patch.clone() for _ in range(per_row_limit)]
            row_fixed = torch.cat(patches, dim=2)
        else:
            c, h, w = maybe_row.shape
            # 拆分 patch 并补灰
            patch_w = w // per_row_limit if w >= per_row_limit else w
            patches = [maybe_row[:, :, j*patch_w:(j+1)*patch_w] for j in range(per_row_limit)]
            while len(patches) < per_row_limit:
                patches.append(torch.zeros_like(patches[0]))
            row_fixed = torch.cat(patches, dim=2)

            # 若行高度与参考不同，则 resize 到一致
            if row_fixed.shape[1] != ref_h:
                row_fixed = F.interpolate(
                    row_fixed.unsqueeze(0), size=(ref_h, row_fixed.shape[2]), mode="bilinear", align_corners=False
                ).squeeze(0)

        # 缩略图 resize 匹配行高
        thumb = F.interpolate(thumbs[i].unsqueeze(0), size=(ref_h, ref_h), mode="bilinear", align_corners=False).squeeze(0)
        row = torch.cat([thumb, row_fixed], dim=2)
        padded_rows.append(row)

    # --- 对齐行宽 ---
    max_width = max(r.shape[2] for r in padded_rows)
    aligned_rows = []
    for r in padded_rows:
        pad_right = max_width - r.shape[2]
        if pad_right > 0:
            r = F.pad(r, (0, pad_right, 0, 0), value=0.5)
        aligned_rows.append(r)

    grid_full = torch.cat(aligned_rows, dim=1)

    # --- 绘图 ---
    plt.figure(figsize=(12, len(row_grids) * 2))
    plt.imshow(grid_full.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title(f"Decoded patches for token {token_id}\n灰色行表示该图未出现此 token", fontsize=10)
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"✅ 保存可视化结果: {save_path}")
    
    
# image_path = [
#     "/home/110/u110003/code/1007/SimVQ-main/configs/source/2.png",
#     "/home/110/u110003/code/1007/SimVQ-main/configs/source/3.png",
#     "/home/110/u110003/code/1007/SimVQ-main/configs/source/4.png",
#     "/home/110/u110003/code/1007/SimVQ-main/configs/source/5.png",
#     "/home/110/u110003/code/1007/SimVQ-main/configs/source/6.png",
# ]




def _filter_state_dict(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    out = {}
    plen = len(prefix)
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[plen:]] = v
    return out


def main(args):
    ddconfig = dict(
        double_z=False, z_channels=128, resolution=128,
        in_channels=3, out_ch=3, ch=128,
        ch_mult=[1, 2, 2, 4], num_res_blocks=2
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.method, exist_ok=True)
    
    print("Loading ckpt from: ", args.ckpt_path)
    raw_ckpt = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = raw_ckpt.get("state_dict", raw_ckpt)

    # ===== 构建模型 =====
    if args.method == 'cyclemlp':
        layers = [3, 4, 24, 3]
        mlp_ratios = [4, 4, 4, 4]
        embed_dims = [64, 96, 128, 128]
        transitions = [True, True, False, False]
        patch_size, patch_stride = 3, 2
        encoder = CycleNet(layers, embed_dims=embed_dims, patch_size=patch_size,
                           transitions=transitions, mlp_ratios=mlp_ratios,
                           mlp_fn=CycleMLP, patch_stride=patch_stride)
        encoder.default_cfg = default_cfgs['cycle_L']
    elif args.method == 'convmlp':
        encoder = convmlp_l_encoder()
    else:
        encoder = Encoder(**ddconfig)
            
    encoder.load_state_dict(_filter_state_dict(state_dict, "encoder."), strict=True)

    quantize = VectorQuantizer(1024, 128, beta=0.25)
    quantize.embedding.weight.data.copy_(state_dict["quantize.embedding.weight"])
    quantize.embedding_proj.weight.data.copy_(state_dict["quantize.embedding_proj.weight"])
    
    decoder = Decoder(**ddconfig)
    decoder.load_state_dict(_filter_state_dict(state_dict, "decoder."), strict=True)

    encoder.eval().to(device)
    quantize.eval().to(device)
    decoder.eval().to(device)

    imagenet_root = "/home/common/ImageNet/ImageNet2012/train"
    image_path = get_random_imagenet_images(imagenet_root, num_classes=20, per_class=1, save_dir=f"{args.method}/samples_20x2")

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # =====================
    # Step 1️⃣ 统计所有 token 的出现次数
    # =====================
    token_counter = Counter()
    per_image_token_indices = {}  # 每张图的 token 索引列表

    for img in image_path:
        img_name = os.path.basename(img)
        x = Image.open(img).convert("RGB")
        x = tfm(x).unsqueeze(0).to(device)
        metric_mode = args.metric.lower()

        with torch.no_grad():
            h = encoder(x)
            print(f"latent shape: {h.shape}")
            emb_enc = quantize.embedding_proj(quantize.embedding.weight)

            if metric_mode == "cosine":
                h_for_assign = F.normalize(h, p=2, dim=1)
                enc_for_assign = F.normalize(emb_enc, p=2, dim=1)
            else:
                h_for_assign = h
                enc_for_assign = emb_enc

            h_flat = rearrange(h_for_assign, "b c h w -> (b h w) c")

            # 匹配 token
            if metric_mode == "euclidean":
                dists = torch.cdist(h_flat, enc_for_assign, p=2)
                min_encoding_indices = torch.argmin(dists, dim=1)
            else:
                sims = torch.matmul(h_flat, enc_for_assign.t())
                _, min_encoding_indices = torch.max(sims, dim=1)

            # 更新计数
            token_counter.update(min_encoding_indices.cpu().tolist())
            per_image_token_indices[img_name] = min_encoding_indices.cpu()

        print(f"✅ {img_name}: 共分配 {len(min_encoding_indices)} 个 token")  # 32x32
    

    top_k = args.top_k
    most_common_tokens = token_counter.most_common(8 * top_k)
    candidate_tokens = [t for t, _ in most_common_tokens]

    # random.seed(42)
    selected_tokens = random.sample(candidate_tokens, k=min(top_k, len(candidate_tokens)))

    print(f"\n🎲 从前 {8 * top_k} 个高频 token 中随机选择 {top_k} 个:")
    for t in selected_tokens:
        print(f"  - token {t}: {token_counter[t]} 次")

    # ========================= 可视化部分 =========================
    for img in image_path:
        img_name = os.path.basename(img)
        token_indices = per_image_token_indices[img_name]
        h_feat = w_feat = int(math.sqrt(len(token_indices)))

        for target_token in selected_tokens:
            draw_token_boxes_on_image(
                image_path=img,
                token_indices=token_indices,
                target_token=target_token,
                h_feat=h_feat,
                w_feat=w_feat,
                save_dir=f"{args.method}/token_boxes"
            )

    # ========================= 解码/可视化行列 =========================
    per_image_limit = 5
    for target_token in selected_tokens:
        count = token_counter[target_token]
        print(f"\n==============================")
        print(f"🎯 可视化 token {target_token}（出现 {count} 次）")
        print(f"==============================")

        row_grids = []  # 每张图一行（即使该图没有该token，也要append None）

        for img in image_path:
            img_name = os.path.basename(img)
            token_indices = per_image_token_indices[img_name]
            x = Image.open(img).convert("RGB")
            x = tfm(x).unsqueeze(0).to(device)

            with torch.no_grad():
                h = encoder(x)
                h_flat = rearrange(h, "b c h w -> (b h w) c")

                selected = h_flat[token_indices == target_token]
                if selected.numel() > 0:
                    n_select = min(per_image_limit, selected.shape[0])
                    indices = torch.randperm(selected.shape[0])[:n_select]
                    sampled = selected[indices].to(device)

                    # 解码这些 patch
                    imgs = []
                    for i in range(sampled.shape[0]):
                        patch = sampled[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                        img_dec = decoder(patch)
                        # print(f"{img_dec.shape=}")
                        imgs.append(img_dec)
                    imgs = torch.cat(imgs, dim=0)

                    if imgs.shape[0] < per_image_limit:
                        patch_h, patch_w = imgs.shape[2], imgs.shape[3]
                        pad_n = per_image_limit - imgs.shape[0]
                        placeholder = torch.zeros((pad_n, 3, patch_h, patch_w)) * 0.5  # 灰色补丁
                        imgs = torch.cat([imgs, placeholder.to(device)], dim=0)
                        
                    # 每行是一个图片的结果
                    grid_row = make_grid(
                        imgs, nrow=per_image_limit, normalize=True, value_range=(-1, 1)
                    )
                    row_grids.append(grid_row.cpu())
                    print(f"✅ {img_name}: 采样 {n_select} 个 patch")
                else:
                    print(f"⚠️ {img_name}: 未找到该 token，使用占位行代替")
                    placeholder = torch.zeros((1, 3, 8, 8)) 
                    placeholder_row = torch.cat([placeholder for _ in range(per_image_limit)], dim=0)
                    grid_row = make_grid(
                        placeholder_row, nrow=per_image_limit, normalize=True, value_range=(-1, 1)
                    )
                    row_grids.append(grid_row.cpu())

        save_path = f"{args.method}/token_{target_token}_rows.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        visualize_token_rows(row_grids, image_path, target_token, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-select top-K tokens for visualization.")
    parser.add_argument('--metric', type=str, default="cosine",
                        choices=["euclidean", "dot", "cosine"],
                        help="distance metric for quantization")
    parser.add_argument('--ckpt_path', type=str,
                        default="/home/110/u110003/code/1007/SimVQ-main/compare/linear-div_convmlp_6.75M_noconn_/epoch=44-step=472000.ckpt")
    parser.add_argument('--method', type=str, default="convmlp",
                        choices=["cyclemlp", "convmlp", "sim"])
    parser.add_argument('--top_k', type=int, default=10,
                        help="number of most frequent tokens to visualize")
    args = parser.parse_args()
    main(args)


'''
srun --gpus=1 --partition=gpu-3090-01 --ntasks-per-node=1 python draw_cluster.py --ckpt_path /home/110/u110003/ckpt/simvq/imagenet_1024/epoch=49-step=250250.ckpt --method sim

srun --gpus=1 --partition=gpu-3090-01 --ntasks-per-node=1 python draw_cluster.py --ckpt_path /home/110/u110003/code/1007/SimVQ-main/compare/linear-div_convmlp_6.75M_noconn_/epoch=44-step=472000.ckpt --method convmlp

srun --gpus=1 --partition=gpu-3090-01 --ntasks-per-node=1 python draw_cluster.py --ckpt_path /home/110/u110003/code/1007/SimVQ-main/compare/linear-div_cycle_6.75M_noconn/epoch=44-step=473500.ckpt --method cyclemlp
'''