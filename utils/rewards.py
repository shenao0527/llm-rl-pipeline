"""
utils/rewards.py — Rule-based Reward Functions for GSPO RL Training

三组 rule-based 奖励函数，无需训练 reward model:
  1. format_reward     — 格式校验（检查 <answer> 标签、think 块等）
  2. anomaly_reward    — 异常检测（惩罚截断、乱码、重复等）
  3. correctness_reward — 正确性评估（数值、选项、数学表达式匹配）

参考: Jackrong-llm-finetuning-guide 的 reward function 设计
"""

import re
from typing import Optional


def extract_answer(text: str, pattern: str = "auto") -> Optional[str]:
    """
    从模型输出中提取最终答案。

    Args:
        text: 模型生成的完整文本
        pattern: 提取模式
            - "answer": 提取 <answer>...</answer> 标签内容
            - "boxed":  提取 \\boxed{...} 内容
            - "final":  提取 "答案是/答案为/Therefore" 后的内容
            - "auto":   依次尝试以上模式

    Returns:
        提取到的答案字符串，或 None
    """
    text = text.strip()

    extractors = {
        "answer": lambda t: _extract_tag(t, "answer"),
        "boxed":  lambda t: _extract_latex_boxed(t),
        "final":  lambda t: _extract_final_statement(t),
    }

    if pattern == "auto":
        for pat in ["answer", "boxed", "final"]:
            result = extractors[pat](text)
            if result is not None:
                return result
        return None
    else:
        return extractors.get(pattern, lambda t: None)(text)


def _extract_tag(text: str, tag: str) -> Optional[str]:
    """提取 <tag>...</tag> 中的内容"""
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _extract_latex_boxed(text: str) -> Optional[str]:
    """提取 \\boxed{...} 中的内容"""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    return None


def _extract_final_statement(text: str) -> Optional[str]:
    """提取 "答案是/答案为/Therefore" 等之后的内容"""
    patterns = [
        r"答案是[：:]\s*(.+)",
        r"答案为[：:]\s*(.+)",
        r"最终答案[为是][：:]\s*(.+)",
        r"(?:Therefore|Thus|Hence|So|The answer is)[,:]?\s*(.+)",
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


# ============================================================
# Reward 1: Format Reward — 格式校验
# ============================================================

def format_reward(completions: list[str], **kwargs) -> list[float]:
    """
    检查输出格式是否符合要求。

    打分规则（每条独立）:
      + 0.25  — 包含 <think>...</think> 块 (CoT 推理)
      + 0.25  — 包含 <answer>...</answer> 标签
      + 0.25  — <answer> 内容非空
      + 0.25  — 没有多余的 <answer> 标签（重复标签）
      → 总分 0.0 ~ 1.0

    Args:
        completions: 模型生成的 completion 列表

    Returns:
        每条 completion 的格式分数
    """
    rewards = []
    for text in completions:
        score = 0.0

        # 1. 检查 think 块
        m1 = re.search(r" thinking", text, re.IGNORECASE)
        m2 = re.search(r" response", text, re.IGNORECASE)
        if m1 and m2:
            score += 0.25

        # 2. 检查 answer 标签
        answer_matches = re.findall(r"<answer>", text, re.IGNORECASE)
        close_matches = re.findall(r"</answer>", text, re.IGNORECASE)

        if len(answer_matches) >= 1 and len(close_matches) >= 1:
            score += 0.25

            # 3. answer 内容非空
            answer_content = _extract_tag(text, "answer")
            if answer_content and len(answer_content.strip()) > 0:
                score += 0.25

            # 4. 无多余重复标签
            if len(answer_matches) == 1 and len(close_matches) == 1:
                score += 0.25

        rewards.append(score)
    return rewards


# ============================================================
# Reward 2: Anomaly Reward — 异常检测
# ============================================================

def anomaly_reward(completions: list[str], **kwargs) -> list[float]:
    """
    检测并惩罚异常输出。

    打分规则（每条独立，从 1.0 开始扣分）:
      -0.3  — 输出被截断（以不完整 token 结尾）
      -0.3  — 包含明显乱码/无意义重复 (>5 次重复短语)
      -0.2  — 输出过短 (< 50 字符，说明模型敷衍)
      -0.2  — 输出过长且杂乱 (> 3000 字符，说明模型开始胡言乱语)
      → 总分 0.0 ~ 1.0

    Args:
        completions: 模型生成的 completion 列表

    Returns:
        每条 completion 的异常分数
    """
    rewards = []
    for text in completions:
        score = 1.0

        # 1. 截断检测
        if _is_truncated(text):
            score -= 0.3

        # 2. 乱码/重复检测
        if _has_repetition(text, threshold=5):
            score -= 0.3

        # 3. 过短检测
        if len(text.strip()) < 50:
            score -= 0.2

        # 4. 过长杂乱检测
        if len(text) > 3000:
            score -= 0.2

        rewards.append(max(0.0, score))
    return rewards


def _is_truncated(text: str) -> bool:
    """检查文本是否以不完整 token 结尾"""
    # 以未闭合的标签、不完整的特殊字符结尾
    truncated_patterns = [
        r"<[^>]*$",             # 未闭合标签
        r"[a-zA-Z]{1,2}$",      # 截断的英文单词 (大概率)
        r"\\[a-zA-Z]+$",        # 截断的 LaTeX
    ]
    for pat in truncated_patterns:
        if re.search(pat, text.strip()):
            return True
    return False


def _has_repetition(text: str, threshold: int = 5) -> bool:
    """检查是否存在重复短语（长度 ≥ 10 字符，重复 ≥ threshold 次）"""
    # 滑动窗口检测重复
    for win_len in range(10, min(50, len(text) // threshold + 1)):
        for i in range(0, len(text) - win_len * threshold, win_len):
            chunk = text[i:i + win_len]
            count = text.count(chunk)
            if count >= threshold:
                return True
    return False


# ============================================================
# Reward 3: Correctness Reward — 正确性评估
# ============================================================

def correctness_reward(
    completions: list[str],
    solution: list[str],
    match_type: str = "auto",
    **kwargs
) -> list[float]:
    """
    将模型答案与正确答案匹配，评估正确性。

    匹配策略:
      - numeric: 提取数字，比较差异 < 1e-2
      - option:  提取 A/B/C/D 选项，精确匹配
      - math:    提取数学表达式，用 sympy 比较等价性
      - auto:    自动选择策略

    Args:
        completions: 模型生成的 completion 列表
        solution:    正确答案列表（长度与 completions 一致）
        match_type:  匹配策略

    Returns:
        每条 completion 的正确性分数 (0.0 或 1.0)
    """
    if len(completions) != len(solution):
        raise ValueError(
            f"completions ({len(completions)}) and solution ({len(solution)}) must have same length"
        )

    rewards = []
    for text, sol in zip(completions, solution):
        pred = extract_answer(text)
        if pred is None:
            rewards.append(0.0)
            continue

        if match_type == "numeric" or (match_type == "auto" and _is_numeric(sol)):
            rewards.append(1.0 if _match_numeric(pred, sol) else 0.0)
        elif match_type == "option" or (match_type == "auto" and _is_option(sol)):
            rewards.append(1.0 if _match_option(pred, sol) else 0.0)
        elif match_type == "math" or (match_type == "auto" and _is_math_expr(sol)):
            rewards.append(1.0 if _match_math(pred, sol) else 0.0)
        else:
            # fallback: 字符串相似度
            rewards.append(1.0 if _match_string(pred, sol) else 0.0)

    return rewards


def _is_numeric(s: str) -> bool:
    """判断字符串是否为数值"""
    s = s.strip().replace(",", "").replace(" ", "")
    try:
        float(s)
        return True
    except ValueError:
        return False


def _is_option(s: str) -> bool:
    """判断是否为选项答案 (A/B/C/D)"""
    return bool(re.match(r"^[A-Da-d]$", s.strip()))


def _is_math_expr(s: str) -> bool:
    """判断是否包含数学表达式"""
    return bool(re.search(r"[+\-*/^\\sqrt\int\sum]", s))


def _match_numeric(pred: str, gold: str, tol: float = 1e-2) -> bool:
    """数值匹配（容忍度 tol）"""
    try:
        p = float(pred.strip().replace(",", "").replace(" ", ""))
        g = float(gold.strip().replace(",", "").replace(" ", ""))
        if g == 0:
            return abs(p) < tol
        return abs(p - g) / abs(g) < tol
    except (ValueError, ZeroDivisionError):
        return False


def _match_option(pred: str, gold: str) -> bool:
    """选项精确匹配（大小写不敏感）"""
    pred_clean = pred.strip().upper().rstrip(".")
    gold_clean = gold.strip().upper().rstrip(".")
    # 支持 "A"、"(A)"、"A." 等格式
    pred_char = re.sub(r"[^A-D]", "", pred_clean)
    gold_char = re.sub(r"[^A-D]", "", gold_clean)
    return pred_char == gold_char and len(pred_char) > 0


def _match_math(pred: str, gold: str) -> bool:
    """数学表达式等价性比较"""
    try:
        import sympy as sp
        p = sp.sympify(pred.strip())
        g = sp.sympify(gold.strip())
        return bool(sp.simplify(p - g) == 0)
    except Exception:
        # sympy 解析失败，回退到字符串匹配
        return _match_string(pred, gold)


def _match_string(pred: str, gold: str) -> bool:
    """字符串相似度匹配（归一化后比较）"""
    def normalize(s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^\w\s.]", "", s)
        return s

    return normalize(pred) == normalize(gold)


# ============================================================
# 组合奖励
# ============================================================

def compute_rewards(
    completions: list[str],
    solution: list[str],
    weights: Optional[dict] = None,
    match_type: str = "auto",
) -> list[float]:
    """
    计算组合奖励（加权和）。

    Args:
        completions: 模型生成的 completion 列表
        solution:    正确答案列表
        weights:     各奖励函数权重，默认 {"format": 0.2, "anomaly": 0.3, "correctness": 0.5}
        match_type:  正确性匹配策略

    Returns:
        每条 completion 的综合分数 (0.0 ~ 1.0)
    """
    if weights is None:
        weights = {"format": 0.2, "anomaly": 0.3, "correctness": 0.5}

    f_rewards = format_reward(completions)
    a_rewards = anomaly_reward(completions)
    c_rewards = correctness_reward(completions, solution, match_type=match_type)

    total = []
    for f, a, c in zip(f_rewards, a_rewards, c_rewards):
        score = (
            weights.get("format", 0.2) * f +
            weights.get("anomaly", 0.3) * a +
            weights.get("correctness", 0.5) * c
        )
        total.append(score)

    return total


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    # 快速自测
    test_completions = [
        "<think>Let me calculate 2+2</think>\n<answer>4</answer>",
        "The answer is 5",
        "<think>Hmm</think><answer>3</answer><answer>4</answer>",
        "aaaaaaaaaaaaaaaaaaaa" * 100,  # 重复
    ]
    test_solutions = ["4", "4", "4", "4"]

    print("=== Format Rewards ===")
    for i, r in enumerate(format_reward(test_completions)):
        print(f"  [{i}] score={r:.2f}  |  {test_completions[i][:60]}...")

    print("\n=== Anomaly Rewards ===")
    for i, r in enumerate(anomaly_reward(test_completions)):
        print(f"  [{i}] score={r:.2f}  |  {test_completions[i][:60]}...")

    print("\n=== Correctness Rewards ===")
    for i, r in enumerate(correctness_reward(test_completions, test_solutions)):
        print(f"  [{i}] score={r:.2f}  |  pred={extract_answer(test_completions[i])} gold={test_solutions[i]}")

    print("\n=== Combined Rewards ===")
    for i, r in enumerate(compute_rewards(test_completions, test_solutions)):
        print(f"  [{i}] total={r:.2f}")