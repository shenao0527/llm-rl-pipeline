"""
utils/schema.py — 数据 Schema 定义与格式化工具

统一 SFT 和 RL 阶段的数据格式，确保从 Jackrong 数据集
到训练 pipeline 的无缝衔接。

数据流:
  HuggingFace Dataset → SFTDataFormatter → ChatML → SFT Training
  HuggingFace Dataset → RLDataFormatter  → prompt+answer → RL Training
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ChatMessage:
    """单条消息"""
    role: str           # "system", "user", "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    def to_chatml(self) -> str:
        """转为 ChatML 格式字符串"""
        return f"<|im_start|>{self.role}\n{self.content}<|im_end|>"


@dataclass
class TrainingSample:
    """SFT/RL 训练的通用数据样本"""
    messages: list[ChatMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_chatml(self) -> str:
        """完整对话转为 ChatML 格式"""
        parts = [msg.to_chatml() for msg in self.messages]
        return "\n".join(parts) + "\n<|im_start|>assistant\n"

    @property
    def prompt(self) -> str:
        """仅保留到 assistant 之前的 prompt（用于 RL 推理）"""
        parts = []
        for msg in self.messages:
            if msg.role == "assistant":
                break
            parts.append(msg.to_chatml())
        return "\n".join(parts) + "\n<|im_start|>assistant\n"

    @property
    def answer(self) -> str:
        """提取 assistant 回答（用于 RL reward 计算）"""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return ""


class SFTDataFormatter:
    """
    SFT 数据格式化器

    将 HuggingFace dataset (messages 格式) 转为 Unsloth 训练的 ChatML 格式。
    支持 Jackrong 数据集格式:
      {
        "messages": [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "..."},
          {"role": "assistant", "content": "..."}
        ]
      }
    """

    SYSTEM_PROMPT = (
        "You are a helpful AI assistant. "
        "Think step by step, and provide your final answer within <answer>...</answer> tags."
    )

    def __init__(self, max_seq_length: int = 2048):
        self.max_seq_length = max_seq_length

    def format(self, example: dict) -> str:
        """将单条 HuggingFace example 转为 ChatML 字符串"""
        messages = example.get("messages", [])

        # 确保有 system prompt
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + list(messages)

        # 转为 ChatML
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        text = "\n".join(parts)

        # 截断过长的样本
        if len(text) > self.max_seq_length * 4:  # 粗略估计
            text = text[:self.max_seq_length * 4]

        return text

    def get_formatting_prompts_func(self):
        """返回 Unsloth 兼容的 formatting function"""
        def _func(examples):
            return {"text": [self.format({"messages": ex}) for ex in examples["messages"]]}
        return _func


class RLDataFormatter:
    """
    RL 训练数据格式化器

    将 HuggingFace dataset 转为 (prompt, answer) 对，
    用于 TRL GRPOTrainer 的 reward 计算。
    """

    def __init__(self, max_prompt_length: int = 1536):
        self.max_prompt_length = max_prompt_length

    def format(self, example: dict) -> TrainingSample:
        """将单条 example 转为 TrainingSample"""
        messages = example.get("messages", [])
        sample = TrainingSample()

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            sample.messages.append(ChatMessage(role=role, content=content))

        sample.metadata = {k: v for k, v in example.items() if k != "messages"}
        return sample

    def format_batch(self, examples: dict) -> dict:
        """批量格式化"""
        prompts = []
        answers = []
        for i in range(len(examples.get("messages", []))):
            sample = self.format({"messages": examples["messages"][i]})
            prompts.append(sample.prompt)
            answers.append(sample.answer)

        return {"prompt": prompts, "answer": answers}


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    # 模拟 Jackrong 数据集格式
    test_example = {
        "messages": [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": " thinkingLet me calculate: 2+2=4 response\n<answer>4</answer>"},
        ]
    }

    # SFT 格式化
    sft = SFTDataFormatter()
    print("=== SFT Formatted ===")
    print(sft.format(test_example)[:300])
    print("...")

    # RL 格式化
    rl = RLDataFormatter()
    sample = rl.format(test_example)
    print("\n=== RL Prompt ===")
    print(sample.prompt[:200])
    print("\n=== RL Answer ===")
    print(sample.answer[:200])