from pathlib import Path

template_path = Path("test-grpo-1.7b-gsm8k/run_qwen2_5-1.7b_math_megatron_template.sh")

def generate(
        vllm_use_v1: str="1",
        use_fused_kernels: str ="True",
        filter_overlong_prompts: str ="True",
        use_kl_loss: str ="True",
        output_dir: Path =Path("test-grpo-1.7b-gsm8k/")):
    def get_exp_name():
        return f"vllmv1_{vllm_use_v1}_fused_{use_fused_kernels}_filter_{filter_overlong_prompts}_ukl_{use_kl_loss}"
    template = template_path.read_text()
    exp_name = f"vllmv1_{vllm_use_v1}_fused_{use_fused_kernels}_filter_{filter_overlong_prompts}_ukl_{use_kl_loss}"
    output = template.format(
        vllm_use_v1=vllm_use_v1,
        use_fused_kernels=use_fused_kernels,
        filter_overlong_prompts=filter_overlong_prompts,
        use_kl_loss=use_kl_loss,
        experiment_name=get_exp_name()
    )
    output_path = output_dir / f"{exp_name}.sh"
    output_path.write_text(output)
    print(f"Generated config at: {output_path}")

if __name__ == "__main__":
    generate()
    generate(vllm_use_v1="0")
    generate(use_fused_kernels="False")
    generate(filter_overlong_prompts="False")
    generate(use_kl_loss="False")