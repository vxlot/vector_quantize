#### uv

##### 新建环境

1. 进入新建的环境文件夹

   ```bash
   cd /home/110/u110003/uv_env/default
   ```

2. 配置 `pyproject.toml` 文件 

   name = "default" 要和文件夹名字一致

   dependencies 列表中手动添加依赖

   保持 torch 和 cuda 版本适配

   ```toml
   [build-system]
   requires = ["setuptools>=68", "wheel"]
   build-backend = "setuptools.build_meta"
   
   [project]
   name = "default"
   version = "0.0.1"
   description = "description of your program"
   requires-python = ">=3.8,<3.9"
   
   dependencies = [
     "torch==1.13.1",
     "torchvision==0.14.1",
     "numpy==1.19.2",
     "albumentations~=1.0.0",
     "opencv-python>=4.2",
     "pudb==2019.2",
     "imageio==2.9.0",
     "imageio-ffmpeg==0.4.2",
     "pytorch-lightning<1.3",
     "omegaconf==2.0.0",
     "test-tube>=0.7.5",
     "streamlit>=0.73.1",
     "einops==0.3.0",
     "more-itertools>=8.0.0",
     "transformers==4.3.1",
   ]
   
   
   [tool.uv]
   index-url = "https://download.pytorch.org/whl/cu118"
   extra-index-url = ["https://pypi.org/simple"]
   ```

3. 安装

   ```bash
   uv sync
   ```

##### 更改新依赖

```bash
uv add "protobuf>=4.21,<5.0"
uv add --editable /home/110/u110003/code/OptVQ  # pip install .
uv add --editable /home/110/u110003/code/1007/non_neg[dali,umap,h5]  # pip install .[dali,umap,h5]
uv add /home/110/u110003/uv_env/pkgs/taming-transformers
uv add git+https://github.com/mit-han-lab/efficientvit.git
uv remove umap
```

关于可编辑安装的setup

```py
install_requires=[
    "torch==1.10.0+cu113",  # 不要在setup里写死，在 pyproject.toml 里指定 index-url = "https://download.pytorch.org/whl/cu113"
    "torchvision==0.11.1",
    ...
],
dependency_links=["https://developer.download.nvidia.com/compute/redist"],
```

##### 激活环境

```bash
source /home/110/u110003/uv_env/default/.venv/bin/activate
deactivate
```

#### git

##### github push

先确认是否有`git`，并在`gitee`新建仓库，进入仓库的管理选项，转移仓库，转移到我所在有权向创建仓库的企业。

```bash
git --version
name:13606398519 pwd:weixiaolu617@gmail
```

如果项目不是clone想来的需要初始化一下

```bash
git init
```

初始化的用add新建分支，有git的用set-url

```bash
git remote add/set-url origin https://gitee.com/dailongquan-cs/simvq-two-linear.git
git checkout -b two-linear-div
```

你可能会用到：

```bash
rm .git/index.lock
ps aux | grep git
pkill -9 git
```

配置

```bash
git config --global user.email "13606398519@163.com"
git config --global user.name "vxlot"
```

在vscode里点加号之后提交

```bash
git push
git push --set-upstream origin master
```

如果需要屏蔽大文件，在`.gitignore`里写文件路径

```git
vq_log
*.pyc
configs/recons
configs/source
compare
```

##### download pkgs

```bash
# 可编辑安装指定版本的包
git clone https://github.com/rwightman/pytorch-image-models
cd pytorch-image-models
git checkout c2ba229d995c33aaaf20e00a5686b4dc857044be
pip install -e .
```

打开浏览器访问：
 👉 https://github.com/rwightman/pytorch-image-models/tree/c2ba229d995c33aaaf20e00a5686b4dc857044be

下载得到一个压缩包：pytorch-image-models-c2ba229.zip

```bash
unzip pytorch-image-models-c2ba229.zip
tar -xvf pytorch-image-models-c2ba229.zip
cd pytorch-image-models-c2ba229d995c33aaaf20e00a5686b4dc857044be
pip install -e .
```

最后验证一下：`python -c "import timm; print(timm.__version__)"`

#### bash

软链接

```bash
ln -s /home/110/u110003/ckpt/dq/64x64_diffusion.pt /home/110/u110003/code/dq/models/64x64_diffusion.pt
```

hf

```python
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="/home/110/u110003/hf_download"
export PYTHONPATH=$(pwd):$PYTHONPATH

ssh 10.10.20.9
source /home/110/u110003/uv_env/muddit/.venv/bin/activate
python -c 'import torch;from diffusers import FluxControlNetModel;FluxControlNetModel.from_pretrained("Xlabs-AI/flux-controlnet-hed-diffusers",torch_dtype=torch.bfloat16,use_safetensors=True)'

srun --gpus=1 --partition=debug-A10-01 --ntasks-per-node=1 python -c 'from diffusers import StableDiffusionPipeline;pipe = StableDiffusionPipeline.from_pretrained("MeissonFlow/Meissonic", trust_remote_code=True)'
python -c 'from diffusers import DiffusionPipeline;pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")'
python -c "from torchvision.datasets import CIFAR100; CIFAR100(root='/home/110/u110003/code/1007/non_neg/data', train=True, download=True)"

python -c "import torch;print(torch.cuda.is_available());print(torch.version.cuda);print(torch.version.git_version);print(torch.backends.cudnn.version())"
ssh-keygen -R 10.10.20.2
```

#### debug

launch.json

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "eval/generate",
            "type": "debugpy",
            "request": "launch",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",  // 决定相对路径文件的起点目录
            "python": "/home/110/u110003/uv_env/seed/.venv/bin/python",
            "program": "generate_embeds.py",  // reconstruct_image.py
            "args": [
                "--config_file=configs/mnist_train_dcae.yaml",
                "--ckpt_path=logs/ckpt/epoch=1-step=17000.ckpt"
            ],
            "env": {
                // "HF_ENDPOINT": "https://hf-mirror.com",
                // "HF_HOME": "/home/110/u110003/hf_download",
                // "CUDA_VISIBLE_DEVICES": "0",
                "PYTHONPATH": "${workspaceFolder}:$PYTHONPATH",  // Python 去哪里找模块，影响 import
            },
        },
    ]
}
```

#### slurm

```bash
#!/bin/bash
#SBATCH --job-name=abc                                #任务名称
#SBATCH --output=/home/110/u110003/slurm/nips/abc/out.log  #输出文件
#SBATCH --error=/home/110/u110003/slurm/nips/abc/err.log  #错误日志文件
#SBATCH --nodes=1                                         #申请节点数量
#SBATCH --partition=debug-4090-01              
#SBATCH --ntasks-per-node=4                               #不用改
#SBATCH --gres=gpu:4                                      #每个节点的GPU
#SBATCH --cpus-per-task=8                                 #每个节点的CPU
#SBATCH --chdir=/home/110/u110003/code/nips/DiffusionDPO      #任务的工作目录

echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
export TORCHELASTIC_ENABLE_FILE_TIMER=1

# 以下写具体的任务
export HF_DATASETS_OFFLINE=1
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="yuvalkirstain/pickapic_v2"

# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~24 hours / 2000 steps

srun --nodes=1 --ntasks-per-node=4 --gres=gpu:4 --cpus-per-task=8 \
    accelerate launch --main_process_port 29415 --mixed_precision="fp16" train.py \
      --pretrained_model_name_or_path=$MODEL_NAME \
      --dataset_name=$DATASET_NAME \
      --train_batch_size=1 \
      --dataloader_num_workers=4 \
      --gradient_accumulation_steps=128 \
      --max_train_steps=2000 \
      --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
      --learning_rate=1e-8 --scale_lr \
      --cache_dir="/home/common/Pick-a-Pic/picapic-v2/" \
      --checkpointing_steps 500 \
      --beta_dpo 5000 \
      --output_dir="./tmp-sd15"
```



```bash
source /home/110/u110003/uv_env/ibq/.venv/bin/activate
sbatch /home/110/u110003/slurm/iclr/train_3090_rqvae.slurm
scancel 319
srun  --gpus=26 gpustat --partition=gpu
squeue --user u110003

srun --gpus=16 --partition=debug-A10-01 --ntasks-per-node=8 python main.py fit --config configs/IBQ/gpu/imagenet_ibqgan_1024.yaml
srun --gpus=6 --partition=gpu-TITAN-01 --ntasks-per-node=8 python main.py fit --config configs/IBQ/gpu/imagenet_ibqgan_1024.yaml

squeue
ssh-keygen -R gpu-node-02
```



#### py tools

##### unpack_npz.py

```python
import numpy as np
from PIL import Image
import os
import argparse


def unpack_npz(npz_path, out_dir, limit=None):
    # 读取 npz 文件
    data = np.load(npz_path)
    images = data["arr"]
    labels = data["label_arr"]

    print("images shape:", images.shape)   # (N, H, W, 3)
    print("labels shape:", labels.shape)   # (N,)

    os.makedirs(out_dir, exist_ok=True)

    num_images = len(images) if limit is None else min(limit, len(images))
    for i in range(num_images):
        img = Image.fromarray(images[i])
        filename = f"{i:05d}_label{labels[i]}.png"
        img.save(os.path.join(out_dir, filename))

    print(f"✅ 已保存 {num_images} 张图片到 {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", default="/tmp/openai-2025-09-25-10-36-02-425281/samples_20x64x64x3.npz")
    parser.add_argument("--out_dir", default="./unpack")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    unpack_npz(args.npz_path, args.out_dir, args.limit)
```

#### py pkgs

##### torchview

```bash
uv add torchview>=0.2.7
```

usage: draw nn.Module's program

```python
from torchview import draw_graph

draw_graph(<your model>, 
                input_data=torch.randn(1, 3, 256, 256, dtype=torch.float16).cuda(), 
                save_graph=False, 
                expand_nested=True
            ).visual_graph.render("vae_graph", format="png", cleanup=True)
```

#### web tools

手绘风格流程图：https://excalidraw.com/

公式识别转latex：https://simpletex.cn

emoji：https://github.com/twitter/twemoji/blob/master/assets/svg/1f60d.svg

#### prompts

我的目标是：将第一个类 (`ImageCaptionLargeDataset`) 中处理和返回 **`prompt_input_ids`** 的逻辑，集成到我正在使用的 **`ImageDataset`** 类中。

加上公式和符号仔细讲讲 公式不要留着markdown$FF^\top$就给我展示了 我要看到公式 就是你渲染好了再给我。

#### diffusion

##### ddpm

reference： https://kexue.fm/archives/9119

扩散模型希望将一张图片 $x_0$ 不断地加噪声最后变成 $x_T \sim \mathcal{N}(\mathbf{0},\boldsymbol{I})$ 学习到 $x_t$ 与 $x_{t-1}$ 的关系，从而推理的时候降噪还原成图像。

$x_t \to p(x_{t-1}|x_t)$ 如果有了这个**分布**，就可以从分布里取一个 $x_{t-1}$ ，让结果更加的具有随机性。
$$
p(x_{t-1}|x_{t})=\frac{p(x_{t}|x_{t-1})p(x_{t-1})}{p(x_{t})}
$$
$p(x_t|x_{t-1})$  是前向加噪过程，是人为设定的，通过设定好的 $\alpha_t\beta_t$ 表进行计算。
$$
{x}_t=\sqrt{\alpha_t}{x}_{t-1}+\sqrt{\beta_t}{\varepsilon}_t,\quad\alpha_t,\beta_t\gt0\quad\alpha_t+\beta_t=1
\quad
\boldsymbol{\varepsilon}_t\sim\mathcal{N}(\mathbf{0},\boldsymbol{I})
$$
因为 $\boldsymbol{\varepsilon}_t\sim\mathcal{N}(\mathbf{0},\boldsymbol{I})$ ，所以 $\sqrt{\beta_t}\boldsymbol{\varepsilon}_t\sim\mathcal{N}(\mathbf{0},\beta_t)$ ，有
$$
p(x_{t}|x_{t-1}) \sim \mathcal{N}(\sqrt{\alpha_t}{x}_{t-1},\beta_t)
$$
另外两项概率单独求解难，考虑加入 $x_0$ 条件，由于马尔可夫性质， $p(x_{t}|x_{t-1},x_0)=p(x_{t}|x_{t-1})$ 
$$
p(x_{t-1}|x_{t},x_0)=\frac{p(x_{t}|x_{t-1})p(x_{t-1}|x_0)}{p(x_{t}|x_0)}
$$
关于 $p(x_{t}|x_0)$ ，有快速采样xt的方法
$$
\begin{align}
{x}_t=\sqrt{\bar{\alpha}_{t}}{x}_0+\sqrt{\bar{\beta}_{t}}\varepsilon,
\quad\bar{\alpha}_{t}={\alpha}_{t}{\alpha}_{t-1}...{\alpha}_{1}
\quad\bar{\alpha}_{t} + \bar{\beta}_{t}=1
\quad{\varepsilon}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})
\end{align}
$$
所以概率上
$$
p(x_{t}|x_0) \sim \mathcal{N}(\sqrt{\bar{\alpha}_{t}}{x}_0,1-\bar{\alpha}_{t})
$$
三个高斯分布做一下运算就得到方差固定的一个分布，**问题转换成求 $\tilde{\mu}(x_0,x_t)$ **。
$$
p(x_{t-1}|x_0,x_t) \sim \mathcal{N}(\tilde{\mu}(x_0,x_t),\tilde{\beta_t})
$$
反向求解就可以从分布中采样：
$$
\begin{align}
{x}_{t-1}=\tilde{\mu}(x_0,x_t) + \sqrt{\tilde{\beta_t}}\varepsilon, 
\quad{\varepsilon}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})
\end{align}
$$
数值上，这个均值的表达式为：
$$
\tilde{\mu}(x_0,x_t)=
\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 +
\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t
$$
此时，如果知道了 $x_0, x_t$ 就可以得到 $x_{t-1}$ 的分布，从而采样出一个  $x_{t-1}$。但是在推理的时候，我们没有 $x_0$ ，所以下面需要重参数化来用 $x_t$ 求出一个可以用来代替的 $x_0$ 。**注意到式（1），变换得到一个工具 $x_0$ **：
$$
x_0=\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t-\sqrt{1-\bar{\alpha}_t}\varepsilon)
$$
这样未知量进一步定位到 $\varepsilon$ ，所以 Unet 输入的是 $x_t$ ，预测一个噪音，**这个噪音是 $t$ 时刻和 $0$ 时刻之间的噪音**。

回顾一下这个路径，$p(x_{t-1}|x_{t}) \to p(x_{t-1}|x_{t},x_0) \to \tilde{\mu}(x_0,x_t) \to x_0 \to \varepsilon$ 。

式（2）最终化简为：
$$
\begin{align}
{x}_{t-1}=\frac{1}{\sqrt{\alpha_t}}x_t - \frac{\beta_t}{\sqrt{\alpha_t(1-\bar{\alpha}_t)}}\varepsilon_\theta(x_t,t) + \sqrt{\tilde{\beta_t}}\varepsilon, 
\quad{\varepsilon}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})
\end{align}
$$
PS: 小例子
$$
\begin{align*}
x_{3} &= \alpha_{3} x_{2} + \beta_{3} \epsilon_{3} \\
    &= \alpha_{3}(\alpha_{2}x_{1} + \beta_{2}\epsilon_{2}) + \beta_{3} \epsilon_{3} \nonumber \\
    &= \alpha_{3}(\alpha_{2}(\alpha_{1}x_{0} + \beta_{1}\epsilon_{1}) + \beta_{2}\epsilon_{2}) + \beta_{3} \epsilon_{3} \nonumber \\
    &= (\alpha_{3}\alpha_{2}\alpha_{1})x_{0} + (\alpha_{3}\alpha_{2})\beta_{1}\epsilon_{1} + \alpha_{3}\beta_{2}\epsilon_{2} + \beta_{3} \epsilon_{3} \nonumber
\end{align*}
$$
右边可以看成多个相互独立的正态噪声之和

正态分布的叠加性：多个独立的正态噪声之和的分布，实际上是均值为0、方差为 $S3$ 的正态分布
$$
\begin{align*}
S_3 &= \alpha_{3}^2\alpha_{2}^2\beta_{1}^2 + \alpha_{3}^2\beta_{2}^2 + \beta_{3}^2 \\
    &= \alpha_{3}^2[\alpha_{2}^2\beta_{1}^2 + \beta_{2}^2] + \beta_{3}^2 \\
    &= \alpha_{3}^2[\alpha_{2}^2(1-\alpha_{1}^2) + (1-\alpha_{2}^2)] + \beta_{3}^2 \\
    &= \alpha_{3}^2[1-\alpha_{2}^2\alpha_{1}^2] + \beta_{3}^2 \\
    &= \alpha_{3}^2[1-\alpha_{2}^2\alpha_{1}^2] + (1-\alpha_{3}^2) \\
    &= 1-\alpha_{3}^2\alpha_{2}^2\alpha_{1}^2 \\
    &= 1-\prod_{i=1}^3\alpha_i^2
\end{align*}
$$
所以有xt的简单计算方法
$$
\begin{align}
\boldsymbol{x}_t=(\alpha_t\cdots\alpha_1)\boldsymbol{x}_0+\sqrt{1-(\alpha_t\cdots\alpha_1)^2}\bar{\boldsymbol{\varepsilon}}_t,\quad\bar{\boldsymbol{\varepsilon}}_t\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})
\end{align}
$$

##### cfg

为了和 ddpm 区分， 我们在条件生成使用 $\hat{p}(x_{t-1}|x_t,y)$ 来表示最终目标。

Classifier guidance 是一种采样方法，他希望复用已经训练好的 ddpm 模型，再额外训练一个分类器。


$$
\hat{p}
$$

#### discrete(mask) diffusion



#### representation learning

The task of representation learning is to learn an encoder function $f : \mathbb{R}^d \rightarrow \mathbb{R}^k$ that extracts low-dimensional data representations $z \in \mathbb{R}^k$ *(a.k.a. features)* from inputs $x \in \mathbb{R}^d$.

Through a simple reparameterization, NCL can remarkably enhance the feature interpretability, sparsity, orthogonality, and disentanglement.

##### 重建式 

**目标**：通过“重建输入”来迫使模型学习有信息量的表示。

**Autoencoder (AE)**：经典做法，压缩 → 重建。

**变分自编码器 (VAE)**：在 AE 基础上加上概率分布约束，表示服从潜在分布。

**Masked Autoencoder (MAE)** / **Masked Diffusion Models**：遮掉一部分输入（像素/patch/噪声）让模型去预测缺失部分，从而学到全局结构感知的表示。

##### 对比式/自监督

**目标**：拉近“正样本对”表示，拉远“负样本对”。

- **SimCLR**、**MoCo**、**BYOL**、**SwAV**：图像对比学习的代表。
- **CLIP**：跨模态对比学习（图像 ↔ 文本）。
- **SimSiam**：去掉负样本，只做正样本一致性。
- **对比蒸馏 (Distillation Contrastive)**：用大模型特征作为教师信号。

##### 监督 / 半监督对齐

**目标**：通过标签或弱监督进行表征对齐。

- **Metric Learning / Triplet Loss**：通过距离约束学习判别表示。
- **多任务学习**：让表示服务于多个任务，从而更通用。
- **半监督表示学习**：少量标签 + 大量无标签数据，典型如 FixMatch、UDA。

##### 基于聚类 / 原型

**目标**：通过聚类一致性来学表示。

- **DeepCluster**：先做聚类，把聚类结果当伪标签训练网络。
- **SwAV**：同时学习聚类中心和对比表示。
- **ProtoNets**（few-shot）：基于类原型向量来对齐表示。

##### 生成式

**目标**：通过生成建模学到表示。

- **生成对抗网络 (GANs)**：判别器和生成器对抗，判别器的特征可以作为表示。
- **扩散模型 (Diffusion Models)**：学习去噪的过程，其隐空间特征可以当作表征。
- **Flow-based Models**（正规化流）：学习可逆的变换，隐变量即为表示。





lipman_flow_2023

liu_flow_2023

#### US签证

https://ceac.state.gov/genniv/

Application ID: AA00F1TQGF  **AA00F30XCR**



Inspired by how large language models are fine-tuned with human feedback, diffusion models have recently adopted similar preference alignment strategies. However, existing approaches like Diffusion-DPO still rely on an imperfect SFT model, which may limit their ability to fully capture human preferences.

https://github.com/twitter/twemoji/blob/master/assets/svg/1f60d.svg









welcome everyone.

i am excited to present our reasearch on ...

first, we breifly show
