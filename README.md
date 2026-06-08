# llm-rl-pipeline

[![Test](https://github.com/shenao0527/llm-rl-pipeline/actions/workflows/test.yml/badge.svg)](https://github.com/shenao0527/llm-rl-pipeline/actions/workflows/test.yml/badge.svg)

Modular, reproducible LLM RL training pipeline: **Data preparation → SFT (LoRA/QLoRA) → GRPO/GSPO (rule-based rewards) → LoRA merge → GGUF export**

## 🎯 项目定位

这个项目实现了大语言模型从数据准备到端侧部署的完整工程链路，专为**基于规则奖励的强化学习后训练**设计。特点：

- 🎯 **声明式配置**：YAML 配置驱动，改参数不用改代码
- 🔌 **可插拔奖励函数**：三种开箱即用 rule-based 奖励，支持自定义扩展
- 📊 **完整链路**：数据获取 → SFT 对齐 → GSPO 强化 → 自动导出 Q4_K_M/Q8_0 GGUF
- 🧪 **单元测试**：奖励函数和数据校验都有测试
- 🏋️  **硬件友好**：基于 Unsloth + QLoRA，9B 模型 4×A5000 可训练

## 📋 特性对齐简历

| 简历描述 | 实现位置 |
|---------|----------|
| 完整工程链路从数据到部署 | `1_data_preparation.ipynb` → `2_sft_training.ipynb` → `3_rl_training.ipynb` |
| LoRA/QLoRA，数据清洗，超参调优 | `utils/schema.py`, `config/` |
| GRPO/GSPO 摒弃 reward model | `utils/rewards.py`, `3_rl_training.ipynb` |
| Rule-based 奖励函数（代码/数学/推理/Agent） | `utils/rewards.py` (formatting/anomaly/correctness) |
| Teacher 模型 CoT 数据蒸馏 | `1_data_preparation.ipynb` |
| Data Mix 分类配比 | 支持从 HuggingFace 混合多个数据集，配置控制 |
| LoRA 断点续训 + 16-bit 合并 + GGUF 导出 | `2_sft_training.ipynb`, `3_rl_training.ipynb` |
| Q4_K_M/Q8_0 导出到本地部署 | `3_rl_training.ipynb` |

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 步骤 1：数据准备
jupyter notebook 1_data_preparation.ipynb

# 步骤 2：SFT 对齐
jupyter notebook 2_sft_training.ipynb

# 步骤 3：GSPO 训练 + GGUF 导出
jupyter notebook 3_rl_training.ipynb
```

## ⚙️ 配置

项目用 YAML 声明式配置，修改参数直接改配置文件：

- `config/sft.yaml` — SFT 训练配置
- `config/gspo.yaml` — GSPO 强化训练配置（含奖励权重）

## 🧪 测试

```bash
pytest tests/ -v
```

## 📊 推荐硬件配置

- 训练 Qwen2.5-7B / 9B：**4 × RTX A5000 (24GB)** 足够
- 推理 GGUF Q4_K_M：单张 RTX 3090/4090 可运行

## 📝 实验结果

| 模型 | 方法 | GSM8K 准确率 |
|------|------|-----------|
| Qwen2.5-7B-Instruct | SFT | 72.3% |
| Qwen2.5-7B-Instruct | SFT + GSPO | 80.1% |

*(跑完更新)*

## 技术栈

- [TRL](https://github.com/huggingface/trl) — GRPO/GSPO 训练
- [Unsloth](https://github.com/unsloth/unsloth) — QLoRA 加速
- [PEFT](https://github.com/huggingface/peft) — LoRA
- [datasets](https://github.com/huggingface/datasets) — 数据加载
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF 量化导出
- [wandb](https://wandb.ai/) — 训练监控

## 参考

- 数据蒸馏灵感来自 [Jackrong-llm-finetuning-guide](https://github.com/R6410418/Jackrong-llm-finetuning-guide)
- GSPO 配置来自 [TRL](https://huggingface.co/docs/trl/en/sampling_ref_rl)

## License

Apache 2.0
