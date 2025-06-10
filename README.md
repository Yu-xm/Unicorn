# Unicorn: Text-Only Data Synthesis for Vision Language Model Training

📄 [Paper](https://arxiv.org/abs/2503.22655) | 🤗 [Data](https://huggingface.co/datasets/BoyaWu10/Bunny-v1_1-data) | 🤖 [Data](https://www.modelscope.cn/datasets/BoyaWu10/Bunny-v1.1-data) | 🤗 [HFSpace](https://huggingface.co/spaces/BoZhaoHuggingFace/Bunny) 🐰 [Demo](http://bunny.baai.ac.cn)

## News

- [2025/06/10] 🔥 We release the **training code** of the **Unicorn**. Try training!
- [2025/04/15] 🔥 Release **Unicorn-1.2M** & **Unicorn-Instruction-471K** Datasets. [[HF](https://huggingface.co/datasets/Yu2020/Unicorn)]

## Training

**Our code is based on** [Bunny](https://github.com/BAAI-DCAI/Bunny).

### Env

Create a conda virtual environment and activate it:

  ```shell
  conda create -n bunny python=3.10
  conda activate bunny
  ```

Basic requirements

  ```shell
  pip install --upgrade pip  # enable PEP 660 support
  pip install transformers
  pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu118
  ```

Install apex

  ```shell
  # https://github.com/NVIDIA/apex#from-source
  pip install ninja
  git clone https://github.com/NVIDIA/apex
  cd apex
  # if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
  # otherwise
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
  ```

Install flash-attention

  ```shell
  # https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
  pip install packaging
  pip install flash-attn --no-build-isolation
  ```

Install bunny and other requirements

  ```shell
  git clone https://github.com/BAAI-DCAI/Bunny.git
  cd Bunny
  pip install -e .
  ```

### Data

Download the **Unicorn-1.2M** & **Unicorn-Instruction-471K** Datasets. [[HF](https://huggingface.co/datasets/Yu2020/Unicorn)]

Embed the captions from **Unicorn-1.2M** to get text embeddings

```
python image_embed.py
```

Mean shift to get synthetic image embeddings

```
python embed_mean.py
```

Then, change the embedding path in `data_utils.py`

```
folder_path = ''
```
Note: the same pkl file is used in both the pretraining and instruction-tuning stages

### Pretrain

```
sh script/train/pretrain.sh
```

### Instruction Tuning

```
sh script/train/finetune_full.sh
```

## Contact

If you have any questions, please contact: xmyu02@gamil.com
