# Just-RL reproduction

**Step 1: download data**

Download data `aime-2024.parquet` and `dapo-math-17k.parquet`:

`bash recipe/dapo/prepare_dapo_data.sh`

This will download data to `/root/verl/data/`:

```txt
root@msra-ep12:/workspaces/verl# ls /root/verl/data/
aime-2024.parquet  dapo-math-17k.parquet
```

**Step 2: download model**

```bash
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
--local-dir $HOME/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

This will download model to ``:

```txt
root@msra-ep12:/workspaces/verl# ls /root/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/
LICENSE  README.md  config.json  figures  generation_config.json  model.safetensors  tokenizer.json  tokenizer_config.jso
```

**Step 3: run script**

```bash
bash run_training.sh
```