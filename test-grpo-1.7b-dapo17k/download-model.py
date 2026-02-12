from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    cache_dir="/data/amlt_data/DeepSeek-R1-Distill-Qwen-1.5B",
)
