# VeRL Training Metrics Documentation

This document provides a comprehensive explanation of all training metrics logged during VeRL PPO/GRPO training, including their mathematical definitions and code implementations.

---

## Summary Table

| Metric | Description | Expected Range | Code Location |
|--------|-------------|----------------|---------------|
| **Global Sequence Length Metrics** |
| `global_seqlen/min` | Minimum total sequence length across partitions (before balancing) | >0 | [verl/utils/seqlen_balancing.py](verl/utils/seqlen_balancing.py#L257) |
| `global_seqlen/max` | Maximum total sequence length across partitions (before balancing) | >0 | [verl/utils/seqlen_balancing.py](verl/utils/seqlen_balancing.py#L257) |
| `global_seqlen/minmax_diff` | Difference between max and min (imbalance indicator) | ≥0 (lower is better) | [verl/utils/seqlen_balancing.py](verl/utils/seqlen_balancing.py#L257) |
| `global_seqlen/balanced_min` | Minimum total sequence length across partitions (after balancing) | >0 | [verl/utils/seqlen_balancing.py](verl/utils/seqlen_balancing.py#L257) |
| `global_seqlen/balanced_max` | Maximum total sequence length across partitions (after balancing) | >0 | [verl/utils/seqlen_balancing.py](verl/utils/seqlen_balancing.py#L257) |
| `global_seqlen/mean` | Mean total sequence length per partition | >0 | [verl/utils/seqlen_balancing.py](verl/utils/seqlen_balancing.py#L257) |
| **Actor Metrics** |
| `actor/entropy` | Policy entropy (exploration measure) | ≥0 | [verl/workers/actor/dp_actor.py](verl/workers/actor/dp_actor.py) |
| `actor/pg_loss` | Policy gradient loss | ℝ | [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py#L1160) |
| `actor/kl_loss` | KL divergence penalty loss | ≥0 | [verl/workers/utils/losses.py](verl/workers/utils/losses.py#L163) |
| `actor/pg_clipfrac` | Fraction of samples clipped by PPO upper bound | [0, 1] | [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py#L1227) |
| `actor/ppo_kl` | Approximate KL divergence between old and new policy | ≥0 | [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py#L1214) |
| `actor/pg_clipfrac_lower` | Fraction of negative advantages clipped by lower bound | [0, 1] | [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py#L1231) |
| `actor/grad_norm` | Gradient norm after clipping | ≥0 | [verl/workers/actor/dp_actor.py](verl/workers/actor/dp_actor.py) |
| `actor/lr` | Current learning rate | ≥0 | [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py) |
| **Training Debug Metrics** |
| `training/rollout_probs_diff_valid` | Validity flag (1=valid, 0=invalid) | {0, 1} | [verl/utils/debug/metrics.py](verl/utils/debug/metrics.py#L63) |
| `training/rollout_probs_diff_max` | Max probability difference between rollout and actor | [0, 1] | [verl/utils/debug/metrics.py](verl/utils/debug/metrics.py#L105) |
| `training/rollout_probs_diff_mean` | Mean probability difference | [0, 1] (ideal <0.005) | [verl/utils/debug/metrics.py](verl/utils/debug/metrics.py#L106) |
| `training/rollout_probs_diff_std` | Std of probability difference | ≥0 | [verl/utils/debug/metrics.py](verl/utils/debug/metrics.py#L107) |
| `training/rollout_actor_probs_pearson_corr` | Pearson correlation between rollout and actor probs | [-1, 1] (ideal ≈1) | [verl/utils/debug/metrics.py](verl/utils/debug/metrics.py#L101) |
| `training/global_step` | Current global training step | ≥0 | [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py) |
| `training/epoch` | Current epoch number | ≥0 | [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py) |
| **Rollout Correction Metrics** |
| `rollout_corr/training_ppl` | Training policy perplexity | ≥1 | [verl/trainer/ppo/rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py#L862) |
| `rollout_corr/training_log_ppl` | Log of training policy perplexity | ≥0 | [verl/trainer/ppo/rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py#L865) |
| `rollout_corr/kl` | KL divergence KL(π_rollout ‖ π_training) | ℝ | [verl/trainer/ppo/rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py#L872) |
| `rollout_corr/k3_kl` | K3 KL estimator (more stable for small KL) | ≥0 | [verl/trainer/ppo/rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py#L878) |
| `rollout_corr/rollout_ppl` | Rollout policy perplexity | ≥1 | [verl/trainer/ppo/rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py#L883) |
| `rollout_corr/rollout_log_ppl` | Log of rollout policy perplexity | ≥0 | [verl/trainer/ppo/rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py#L885) |
| `rollout_corr/log_ppl_diff` | Mean log perplexity difference | ℝ | [verl/trainer/ppo/rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py#L893) |
| `rollout_corr/log_ppl_abs_diff` | Mean absolute log perplexity difference | ≥0 | [verl/trainer/ppo/rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py#L894) |
| `rollout_corr/log_ppl_diff_max` | Max log perplexity difference | ℝ | [verl/trainer/ppo/rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py#L895) |
| `rollout_corr/log_ppl_diff_min` | Min log perplexity difference | ℝ | [verl/trainer/ppo/rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py#L896) |
| `rollout_corr/ppl_ratio` | Ratio of training to rollout perplexity | >0 (ideal ≈1) | [verl/trainer/ppo/rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py#L903) |
| `rollout_corr/chi2_token` | Token-level χ² divergence | ≥-1 | [verl/trainer/ppo/rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py#L912) |
| `rollout_corr/chi2_seq` | Sequence-level χ² divergence | ≥-1 | [verl/trainer/ppo/rollout_corr_helper.py](verl/trainer/ppo/rollout_corr_helper.py#L918) |
| **Critic Metrics** |
| `critic/score/mean` | Mean reward score across sequences | ℝ | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L116) |
| `critic/score/max` | Maximum reward score | ℝ | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L117) |
| `critic/score/min` | Minimum reward score | ℝ | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L118) |
| `critic/rewards/mean` | Mean total reward per sequence | ℝ | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L120) |
| `critic/rewards/max` | Maximum total reward | ℝ | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L121) |
| `critic/rewards/min` | Minimum total reward | ℝ | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L122) |
| `critic/advantages/mean` | Mean advantage value | ℝ (ideal ≈0) | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L124) |
| `critic/advantages/max` | Maximum advantage value | ℝ | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L125) |
| `critic/advantages/min` | Minimum advantage value | ℝ | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L126) |
| `critic/returns/mean` | Mean return value | ℝ | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L128) |
| `critic/returns/max` | Maximum return value | ℝ | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L129) |
| `critic/returns/min` | Minimum return value | ℝ | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L130) |
| **Response/Prompt Length Metrics** |
| `response_length/mean` | Mean response length in tokens | >0 | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L174) |
| `response_length/max` | Maximum response length | >0 | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L175) |
| `response_length/min` | Minimum response length | ≥0 | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L176) |
| `response_length/clip_ratio` | Fraction of responses hitting max length | [0, 1] | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L177) |
| `response_length_non_aborted/*` | Same metrics excluding aborted samples | - | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L186) |
| `response/aborted_ratio` | Fraction of aborted (zero-length) responses | [0, 1] | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L191) |
| `prompt_length/mean` | Mean prompt length in tokens | >0 | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L193) |
| `prompt_length/max` | Maximum prompt length | >0 | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L194) |
| `prompt_length/min` | Minimum prompt length | >0 | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L195) |
| `prompt_length/clip_ratio` | Fraction of prompts hitting max length | [0, 1] | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L196) |
| `num_turns/*` | Multi-turn conversation statistics | ≥1 | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L200) |
| **Timing Metrics** |
| `timing_s/gen` | Total generation time in seconds | ≥0 | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L230) |
| `timing_s/reward` | Reward computation time | ≥0 | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L230) |
| `timing_s/old_log_prob` | Old log probability computation time | ≥0 | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L230) |
| `timing_s/adv` | Advantage computation time | ≥0 | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L230) |
| `timing_s/update_actor` | Actor update time | ≥0 | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L230) |
| `timing_s/step` | Total step time | ≥0 | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L230) |
| `timing_s/agent_loop/*` | Agent loop timing statistics | ≥0 | [verl/experimental/agent_loop/agent_loop.py](verl/experimental/agent_loop/agent_loop.py#L966) |
| `timing_per_token_ms/*` | Per-token timing in milliseconds | ≥0 | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L247) |
| **Performance Metrics** |
| `perf/mfu/actor_infer` | Model FLOPS Utilization during inference | [0, 1] | [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py#L1549) |
| `perf/mfu/actor` | Model FLOPS Utilization during training | [0, 1] | [verl/workers/fsdp_workers.py](verl/workers/fsdp_workers.py#L933) |
| `perf/max_memory_allocated_gb` | Peak GPU memory allocated | ≥0 | [verl/workers/fsdp_workers.py](verl/workers/fsdp_workers.py) |
| `perf/max_memory_reserved_gb` | Peak GPU memory reserved | ≥0 | [verl/workers/fsdp_workers.py](verl/workers/fsdp_workers.py) |
| `perf/cpu_memory_used_gb` | CPU memory used | ≥0 | [verl/workers/fsdp_workers.py](verl/workers/fsdp_workers.py) |
| `perf/total_num_tokens` | Total tokens processed in step | ≥0 | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L282) |
| `perf/time_per_step` | Total time per training step | ≥0 | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L283) |
| `perf/throughput` | Tokens per second per GPU | ≥0 | [verl/trainer/ppo/metric_utils.py](verl/trainer/ppo/metric_utils.py#L284) |

---

## Detailed Metric Explanations

### 1. Global Sequence Length Metrics (`global_seqlen/*`)

These metrics measure the load balancing quality across distributed workers. Proper balancing ensures efficient parallel training.

**Location:** [verl/utils/seqlen_balancing.py#L257](verl/utils/seqlen_balancing.py#L257)

#### Mathematical Definitions

For a batch with sequence lengths $\{s_1, s_2, ..., s_n\}$ divided into $k$ partitions:

| Metric | Formula | Description |
|--------|---------|-------------|
| `min` | $\min_{i=1}^{k} \sum_{j \in P_i^{before}} s_j$ | Min partition sum before balancing |
| `max` | $\max_{i=1}^{k} \sum_{j \in P_i^{before}} s_j$ | Max partition sum before balancing |
| `minmax_diff` | $\text{max} - \text{min}$ | Imbalance magnitude |
| `balanced_min` | $\min_{i=1}^{k} \sum_{j \in P_i^{after}} s_j$ | Min partition sum after balancing |
| `balanced_max` | $\max_{i=1}^{k} \sum_{j \in P_i^{after}} s_j$ | Max partition sum after balancing |
| `mean` | $\frac{1}{k} \sum_{i=1}^{n} s_i$ | Mean tokens per partition |

#### Code Implementation

```python
def log_seqlen_unbalance(seqlen_list: list[int], partitions: list[list[int]], prefix):
    k_partition = len(partitions)
    batch_size = len(seqlen_list) // k_partition
    min_sum_seqlen = None
    max_sum_seqlen = None
    total_sum_seqlen = 0

    # Calculate min/max BEFORE balancing (consecutive chunks)
    for offset in range(0, len(seqlen_list), batch_size):
        cur_sum_seqlen = sum(seqlen_list[offset : offset + batch_size])
        if min_sum_seqlen is None or cur_sum_seqlen < min_sum_seqlen:
            min_sum_seqlen = cur_sum_seqlen
        if max_sum_seqlen is None or cur_sum_seqlen > max_sum_seqlen:
            max_sum_seqlen = cur_sum_seqlen
        total_sum_seqlen += cur_sum_seqlen

    # Calculate min/max AFTER balancing (using partition indices)
    balanced_sum_seqlen_list = []
    for partition in partitions:
        cur_sum_seqlen_balanced = sum([seqlen_list[i] for i in partition])
        balanced_sum_seqlen_list.append(cur_sum_seqlen_balanced)
    min_sum_seqlen_balanced = min(balanced_sum_seqlen_list)
    max_sum_seqlen_balanced = max(balanced_sum_seqlen_list)

    return {
        f"{prefix}/min": min_sum_seqlen,
        f"{prefix}/max": max_sum_seqlen,
        f"{prefix}/minmax_diff": max_sum_seqlen - min_sum_seqlen,
        f"{prefix}/balanced_min": min_sum_seqlen_balanced,
        f"{prefix}/balanced_max": max_sum_seqlen_balanced,
        f"{prefix}/mean": total_sum_seqlen / len(partitions),
    }
```

---

### 2. Actor Metrics

These metrics track the policy optimization process.

#### 2.1 `actor/entropy`

**Description:** Policy entropy measuring exploration vs exploitation balance.

**Location:** [verl/workers/actor/dp_actor.py](verl/workers/actor/dp_actor.py)

**Mathematical Formula:**

$$H(\pi) = -\sum_{a} \pi(a|s) \log \pi(a|s)$$

Aggregated using masked mean over response tokens:

$$\text{actor/entropy} = \frac{\sum_{t} H_t \cdot m_t}{\sum_{t} m_t}$$

where $m_t$ is the response mask.

#### 2.2 `actor/pg_loss`

**Description:** Policy gradient loss (PPO clipped objective).

**Location:** [verl/trainer/ppo/core_algos.py#L1160](verl/trainer/ppo/core_algos.py#L1160)

**Mathematical Formula:**

$$L^{PG} = -\mathbb{E}\left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio
- $A_t$ is the advantage estimate
- $\epsilon$ is the clip range

#### 2.3 `actor/ppo_kl`

**Description:** Approximate KL divergence between old and new policy.

**Location:** [verl/trainer/ppo/core_algos.py#L1214](verl/trainer/ppo/core_algos.py#L1214)

**Mathematical Formula:**

$$D_{KL}(\pi_{old} \| \pi) \approx \mathbb{E}\left[\log \pi_{old}(a|s) - \log \pi(a|s)\right]$$

**Code:**
```python
negative_approx_kl = log_prob - old_log_prob
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
```

#### 2.4 `actor/pg_clipfrac`

**Description:** Fraction of samples where the probability ratio was clipped by the upper PPO bound.

**Location:** [verl/trainer/ppo/core_algos.py#L1227](verl/trainer/ppo/core_algos.py#L1227)

**Mathematical Formula:**

$$\text{pg\_clipfrac} = \frac{\sum_t \mathbf{1}[L^{clipped}_t > L^{unclipped}_t] \cdot m_t}{\sum_t m_t}$$

**Code:**
```python
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
```

#### 2.5 `actor/pg_clipfrac_lower`

**Description:** Fraction of **negative advantage** samples clipped by the dual-clip lower bound.

**Location:** [verl/trainer/ppo/core_algos.py#L1231](verl/trainer/ppo/core_algos.py#L1231)

**Mathematical Formula (Dual-Clip PPO):**

$$\text{pg\_clipfrac\_lower} = \frac{\sum_t \mathbf{1}[A_t < 0 \land L^{clip1}_t > L^{lower}_t] \cdot m_t}{\sum_t m_t}$$

where $L^{lower}_t = -A_t \cdot c$ with $c = 3.0$ being the lower clip ratio.

**Code:**
```python
pg_losses3 = -advantages * clip_ratio_c  # clip_ratio_c = 3.0
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
pg_clipfrac_lower = verl_F.masked_mean(
    torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
)
```

#### 2.6 `actor/grad_norm`

**Description:** L2 norm of gradients after gradient clipping.

**Location:** [verl/workers/actor/dp_actor.py](verl/workers/actor/dp_actor.py)

**Mathematical Formula:**

$$\|\nabla_\theta L\|_2 = \sqrt{\sum_i (\nabla_{\theta_i} L)^2}$$

#### 2.7 `actor/kl_loss`

**Description:** KL penalty term added to the policy loss (optional, enabled via config).

**Location:** [verl/workers/utils/losses.py#L163](verl/workers/utils/losses.py#L163)

**Mathematical Formula:**

$$L^{KL} = \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

---

### 3. Training Debug Metrics (`training/*`)

These metrics help diagnose precision mismatches between the rollout engine (e.g., vLLM BF16) and training framework (e.g., FSDP FP32).

**Location:** [verl/utils/debug/metrics.py#L63](verl/utils/debug/metrics.py#L63)

#### 3.1 `training/rollout_probs_diff_*`

**Description:** Statistics of absolute probability differences between rollout and actor policies.

**Mathematical Formula:**

$$\text{diff}_t = |\exp(\log p_{actor,t}) - \exp(\log p_{rollout,t})|$$

- `mean`: $\mathbb{E}[\text{diff}_t]$
- `max`: $\max_t \text{diff}_t$
- `std`: $\text{Std}[\text{diff}_t]$

**Interpretation:** Values above 0.01 indicate precision issues. Ideal is <0.005.

**Code:**
```python
actor_probs = torch.exp(actor_old_log_probs)
rollout_probs = torch.exp(rollout_old_log_probs)
rollout_probs_diff = calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)

def calculate_log_prob_diff(log_probs1, log_probs2, mask):
    full_diff = torch.abs(log_probs1 - log_probs2)
    return torch.masked_select(full_diff, mask)
```

#### 3.2 `training/rollout_actor_probs_pearson_corr`

**Description:** Pearson correlation coefficient between rollout and actor probabilities.

**Mathematical Formula:**

$$\rho = \frac{\text{Cov}(p_{actor}, p_{rollout})}{\sigma_{actor} \cdot \sigma_{rollout}}$$

**Code:**
```python
def pearson_correlation_coefficient(tensor1, tensor2, mask):
    mt1 = torch.masked_select(tensor1, mask)
    mt2 = torch.masked_select(tensor2, mask)
    result = torch.corrcoef(torch.stack([mt1, mt2], dim=0))
    return result[0][1].detach().item()
```

---

### 4. Rollout Correction Metrics (`rollout_corr/*`)

These metrics diagnose off-policy issues between rollout and training policies.

**Location:** [verl/trainer/ppo/rollout_corr_helper.py#L830](verl/trainer/ppo/rollout_corr_helper.py#L830)

**Reference:** [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](https://richardli.xyz/rl-collapse)

#### 4.1 `rollout_corr/training_ppl` and `rollout_corr/rollout_ppl`

**Description:** Perplexity of training and rollout policies respectively.

**Mathematical Formula:**

$$\text{PPL} = \exp\left(-\frac{1}{|T|} \sum_{t \in T} \log \pi(y_t|y_{<t})\right)$$

**Code:**
```python
mean_log_prob_training = verl_F.masked_mean(old_log_prob, response_mask, axis=-1)
training_ppl = torch.exp(-mean_log_prob_training).mean()
```

#### 4.2 `rollout_corr/kl`

**Description:** Direct KL divergence estimator: $D_{KL}(\pi_{rollout} \| \pi_{training})$.

**Mathematical Formula:**

$$D_{KL} = \mathbb{E}_{\pi_{rollout}}\left[\log \pi_{rollout} - \log \pi_{training}\right]$$

**Code:**
```python
metrics["kl"] = verl_F.masked_mean(rollout_log_prob - old_log_prob, response_mask).item()
```

#### 4.3 `rollout_corr/k3_kl`

**Description:** K3 KL estimator (more stable for small KL values).

**Mathematical Formula:**

$$D_{KL}^{K3} = \mathbb{E}\left[e^{\log r} - \log r - 1\right] = \mathbb{E}\left[r - \log r - 1\right]$$

where $r = \frac{\pi_{training}}{\pi_{rollout}}$.

**Code:**
```python
log_ratio = old_log_prob - rollout_log_prob
k3_kl_matrix = torch.exp(log_ratio) - log_ratio - 1
metrics["k3_kl"] = verl_F.masked_mean(k3_kl_matrix, response_mask).item()
```

#### 4.4 `rollout_corr/log_ppl_diff`

**Description:** Sequence-level log perplexity difference.

**Mathematical Formula:**

$$\Delta \log \text{PPL} = \frac{1}{N}\sum_i \left(\bar{\log p}_{rollout,i} - \bar{\log p}_{training,i}\right)$$

where $\bar{\log p}_i = \frac{1}{|T_i|}\sum_{t} \log \pi(y_t)$ for sequence $i$.

**Code:**
```python
log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
metrics["log_ppl_diff"] = log_ppl_diff.mean().item()
```

#### 4.5 `rollout_corr/ppl_ratio`

**Description:** Ratio of training PPL to rollout PPL.

**Mathematical Formula:**

$$\text{PPL\_ratio} = \mathbb{E}\left[\exp(\Delta \log \text{PPL})\right] = \mathbb{E}\left[\frac{\text{PPL}_{training}}{\text{PPL}_{rollout}}\right]$$

**Code:**
```python
ppl_ratio = torch.exp(log_ppl_diff).mean()
```

#### 4.6 `rollout_corr/chi2_token` and `rollout_corr/chi2_seq`

**Description:** χ² (Chi-squared) divergence measuring variance of importance sampling weights.

**Mathematical Formula:**

- Token-level: $\chi^2_{token} = \mathbb{E}[\rho^2] - 1$ where $\rho = \frac{\pi_{training}}{\pi_{rollout}}$
- Sequence-level: $\chi^2_{seq} = \mathbb{E}\left[\left(\prod_t \rho_t\right)^2\right] - 1$

**Code:**
```python
# Token-level
log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
rho_token = torch.exp(log_ratio_safe)
rho_squared_token = rho_token.square()
chi2_token = verl_F.masked_mean(rho_squared_token, response_mask) - 1.0

# Sequence-level
log_ratio_sum = verl_F.masked_sum(log_ratio, response_mask, axis=-1)
log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
rho_squared_seq = torch.exp(2.0 * log_ratio_sum_safe)  # (Π ρ_t)²
chi2_seq = rho_squared_seq.mean() - 1.0
```

---

### 5. Critic Metrics (`critic/*`)

These metrics track reward and value estimation quality.

**Location:** [verl/trainer/ppo/metric_utils.py#L76](verl/trainer/ppo/metric_utils.py#L76)

#### 5.1 Score and Reward Metrics

**Mathematical Formula:**

$$\text{sequence\_score}_i = \sum_{t} r_t^{(i)}$$

where $r_t$ are token-level scores from the reward model.

**Code:**
```python
sequence_score = batch.batch["token_level_scores"].sum(-1)
sequence_reward = batch.batch["token_level_rewards"].sum(-1)

# Filter non-aborted samples
non_aborted_sequence_score = sequence_score[non_aborted_mask]
score_mean = torch.mean(non_aborted_sequence_score).item()
score_max = torch.max(non_aborted_sequence_score).item()
score_min = torch.min(non_aborted_sequence_score).item()
```

#### 5.2 Advantages and Returns

**Mathematical Formula:**

Advantages are typically computed using GAE (Generalized Advantage Estimation):

$$A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.

**Code:**
```python
advantages = batch.batch["advantages"]
returns = batch.batch["returns"]

valid_adv = torch.masked_select(advantages, response_mask)
valid_returns = torch.masked_select(returns, response_mask)

metrics["critic/advantages/mean"] = torch.mean(valid_adv).item()
metrics["critic/returns/mean"] = torch.mean(valid_returns).item()
```

---

### 6. Response and Prompt Length Metrics

**Location:** [verl/trainer/ppo/metric_utils.py#L174](verl/trainer/ppo/metric_utils.py#L174)

#### Mathematical Formulas

```python
def _compute_response_info(batch: DataProto):
    response_length = batch.batch["responses"].shape[-1]
    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]
    
    prompt_length = prompt_mask.sum(-1).float()  # Count of 1s
    response_length = response_mask.sum(-1).float()  # Count of 1s
    return dict(response_mask=response_mask, prompt_length=prompt_length, response_length=response_length)
```

**Clip Ratio Formula:**

$$\text{clip\_ratio} = \frac{1}{N}\sum_i \mathbf{1}[\text{len}_i = \text{max\_len}]$$

---

### 7. Timing Metrics (`timing_s/*` and `timing_per_token_ms/*`)

**Location:** [verl/trainer/ppo/metric_utils.py#L230](verl/trainer/ppo/metric_utils.py#L230)

#### 7.1 Raw Timing (`timing_s/*`)

Direct wall-clock time in seconds for each training phase:

| Metric | Description |
|--------|-------------|
| `timing_s/gen` | Sequence generation time |
| `timing_s/reward` | Reward computation time |
| `timing_s/old_log_prob` | Old policy log-prob computation |
| `timing_s/adv` | Advantage computation time |
| `timing_s/update_actor` | Actor gradient update time |
| `timing_s/step` | Total step time |

#### 7.2 Per-Token Timing (`timing_per_token_ms/*`)

**Mathematical Formula:**

$$\text{timing\_per\_token\_ms} = \frac{\text{timing\_s} \times 1000}{\text{num\_tokens}}$$

Different stages use different token counts:
- `gen`: Response tokens only
- `adv`, `update_actor`: All tokens (prompt + response)

**Code:**
```python
num_tokens_of_section = {
    "gen": num_response_tokens,
    **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
}

timing_per_token = {
    f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
    for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
}
```

#### 7.3 Agent Loop Timing (`timing_s/agent_loop/*`)

**Location:** [verl/experimental/agent_loop/agent_loop.py#L966](verl/experimental/agent_loop/agent_loop.py#L966)

Per-sequence timing statistics for multi-turn agent loops:

```python
timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
```

---

### 8. Performance Metrics (`perf/*`)

#### 8.1 `perf/mfu/actor` and `perf/mfu/actor_infer`

**Description:** Model FLOPS Utilization - ratio of achieved FLOPS to theoretical peak.

**Location:** [verl/workers/fsdp_workers.py#L933](verl/workers/fsdp_workers.py#L933)

**Mathematical Formula:**

$$\text{MFU} = \frac{\text{estimated\_flops} \times \text{ppo\_epochs}}{\text{promised\_flops} \times \text{world\_size}}$$

**Code:**
```python
estimated_flops, promised_flops = self.flops_counter.estimate_flops(
    global_num_tokens, delta_time, images_seqlens=images_seqlens
)
metrics["perf/mfu/actor"] = (
    estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
)
```

#### 8.2 Memory Metrics

**Location:** [verl/workers/fsdp_workers.py](verl/workers/fsdp_workers.py)

```python
metrics["perf/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
metrics["perf/max_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024**3)
metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)
```

#### 8.3 Throughput Metrics

**Location:** [verl/trainer/ppo/metric_utils.py#L282](verl/trainer/ppo/metric_utils.py#L282)

**Mathematical Formula:**

$$\text{throughput} = \frac{\text{total\_num\_tokens}}{\text{time\_per\_step} \times \text{n\_gpus}}$$

**Code:**
```python
def compute_throughout_metrics(batch: DataProto, timing_raw: dict, n_gpus: int):
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }
```

---

## Example Metric Values Interpretation

Based on the log excerpt:

```
step:1
global_seqlen/minmax_diff:707659        # 707K token imbalance before balancing
global_seqlen/balanced_max:9144801      # After balancing, max-min diff is only 27 tokens
global_seqlen/balanced_min:9144774      # (9144801 - 9144774 = 27) - excellent balancing!

actor/entropy:1.056                     # Moderate exploration
actor/pg_loss:0.0198                    # Small policy gradient loss
actor/pg_clipfrac:0.0                   # No clipping occurred (early training)
actor/ppo_kl:0.0                        # No KL divergence (early training)

training/rollout_probs_diff_mean:0.004  # Good (<0.005 threshold) - no precision issues
training/rollout_actor_probs_pearson_corr:0.9997  # Excellent correlation

rollout_corr/kl:0.00077                 # Very small KL - on-policy behavior
rollout_corr/chi2_seq:381.14            # High seq-level χ² - some off-policy variance

critic/score/mean:-0.053                # Average reward near 0
critic/advantages/mean:-0.070           # Slightly negative mean advantage

response_length/mean:8813               # Average ~8.8K tokens per response
response_length/clip_ratio:0.098        # ~10% hit max length

perf/throughput:1072                    # ~1K tokens/sec/GPU
```

---

## References

1. [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
2. [Dual-Clip PPO](https://arxiv.org/pdf/1912.09729)
3. [When Speed Kills Stability](https://richardli.xyz/rl-collapse) - Off-policy correction
4. [Precision Mismatch Paper](https://arxiv.org/pdf/2506.13585) - Pearson correlation for debugging
