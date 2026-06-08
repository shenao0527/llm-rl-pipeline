"""
tests/test_rewards.py — 奖励函数单元测试

测试覆盖:
  1. format_reward — 格式校验
  2. anomaly_reward — 异常检测
  3. correctness_reward — 正确性匹配
  4. compute_rewards — 组合奖励
  5. extract_answer — 答案提取

运行: pytest tests/test_rewards.py -v
"""

import sys
from pathlib import Path

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.rewards import (
    format_reward,
    anomaly_reward,
    correctness_reward,
    compute_rewards,
    extract_answer,
)


class TestFormatReward:
    """测试格式奖励函数"""

    def test_perfect_format(self):
        """完整格式应该得满分 1.0"""
        completions = [
            " thinkingLet me think step by step. response\nThe answer is 42.\n<answer>42</answer>"
        ]
        rewards = format_reward(completions)
        assert rewards[0] == 1.0, f"Expected 1.0, got {rewards[0]}"

    def test_missing_think(self):
        """缺少 think 块扣 0.25"""
        completions = ["The answer is <answer>42</answer>"]
        rewards = format_reward(completions)
        assert rewards[0] == 0.75, f"Expected 0.75, got {rewards[0]}"

    def test_missing_answer_tag(self):
        """缺少 answer 标签得 0.25 (仅 think+response)"""
        completions = [" thinking... response\nThe answer is 42."]
        rewards = format_reward(completions)
        assert rewards[0] == 0.25, f"Expected 0.25, got {rewards[0]}"

    def test_empty_answer(self):
        """空 answer 标签扣分"""
        completions = [" thinkingLet me think response\n<answer></answer>"]
        rewards = format_reward(completions)
        assert rewards[0] == 0.75, f"Expected 0.75, got {rewards[0]}"  # think+response(+.25) + tags(+.25) + no dup(+.25) = 0.75

    def test_duplicate_answer_tags(self):
        """重复 answer 标签扣分"""
        completions = [
            " thinkingHmm response<answer>42</answer><answer>43</answer>"
        ]
        rewards = format_reward(completions)
        assert rewards[0] == 0.75, f"Expected 0.75, got {rewards[0]}"

    def test_multiple_completions(self):
        """多 completion 批量测试"""
        completions = [
            " thinkingLet me think response<answer>42</answer>",
            "Just 42",
        ]
        rewards = format_reward(completions)
        assert len(rewards) == 2
        assert rewards[0] == 1.0
        assert rewards[1] == 0.0


class TestAnomalyReward:
    """测试异常检测奖励函数"""

    def test_normal_output(self):
        """正常输出应得满分"""
        completions = ["This is a perfectly normal response with no issues."]
        rewards = anomaly_reward(completions)
        assert rewards[0] == 1.0, f"Expected 1.0, got {rewards[0]}"

    def test_truncated_output(self):
        """截断输出应被检测"""
        completions = ["This sentence is cut off at the en"]
        rewards = anomaly_reward(completions)
        assert rewards[0] <= 0.7, f"Expected <= 0.7, got {rewards[0]}"

    def test_repetitive_output(self):
        """重复输出应被惩罚"""
        completions = ["hello world " * 100]  # 高度重复
        rewards = anomaly_reward(completions)
        assert rewards[0] <= 0.7, f"Expected <= 0.7, got {rewards[0]}"

    def test_too_short_output(self):
        """过短输出应被惩罚"""
        completions = ["Hi"]
        rewards = anomaly_reward(completions)
        assert rewards[0] <= 0.8, f"Expected <= 0.8, got {rewards[0]}"

    def test_score_range(self):
        """分数应在 [0, 1] 范围内"""
        completions = ["normal text", "ab" * 1000, "short"]
        rewards = anomaly_reward(completions)
        for r in rewards:
            assert 0.0 <= r <= 1.0, f"Score {r} out of range [0, 1]"


class TestCorrectnessReward:
    """测试正确性评估函数"""

    def test_numeric_match_exact(self):
        """精确数值匹配"""
        completions = ["<answer>42</answer>"]
        solutions = ["42"]
        rewards = correctness_reward(completions, solutions)
        assert rewards[0] == 1.0

    def test_numeric_match_approx(self):
        """近似数值匹配 (3.14159 vs 3.14)"""
        completions = ["<answer>3.14159</answer>"]
        solutions = ["3.14"]
        rewards = correctness_reward(completions, solutions, match_type="numeric")
        assert rewards[0] == 1.0, f"Expected 1.0, got {rewards[0]}"

    def test_numeric_mismatch(self):
        """数值不匹配"""
        completions = ["<answer>42</answer>"]
        solutions = ["43"]
        rewards = correctness_reward(completions, solutions)
        assert rewards[0] == 0.0

    def test_option_match(self):
        """选项匹配"""
        completions = ["<answer>B</answer>"]
        solutions = ["B"]
        rewards = correctness_reward(completions, solutions, match_type="option")
        assert rewards[0] == 1.0

    def test_option_match_case(self):
        """选项大小写不敏感"""
        completions = ["<answer>b</answer>"]
        solutions = ["B"]
        rewards = correctness_reward(completions, solutions, match_type="option")
        assert rewards[0] == 1.0

    def test_no_answer_extracted(self):
        """无法提取答案应得 0"""
        completions = ["I don't know the answer."]
        solutions = ["42"]
        rewards = correctness_reward(completions, solutions)
        assert rewards[0] == 0.0


class TestExtractAnswer:
    """测试答案提取函数"""

    def test_extract_answer_tag(self):
        text = "thinking... response\n<answer>42</answer>"
        assert extract_answer(text) == "42"

    def test_extract_boxed(self):
        text = "The answer is \\boxed{3.14}"
        assert extract_answer(text) == "3.14"

    def test_extract_final_chinese(self):
        text = "经过计算，答案是：7"
        assert extract_answer(text) == "7"

    def test_extract_nothing(self):
        text = "No answer here"
        assert extract_answer(text) is None

    def test_extract_auto_priority(self):
        """auto 模式优先匹配 answer 标签"""
        text = " thinking response\n<answer>42</answer>\nAlso \\boxed{43}"
        assert extract_answer(text) == "42"


class TestComputeRewards:
    """测试组合奖励函数"""

    def test_combined_weights(self):
        completions = [
            " thinkingLet me think response\n<answer>42</answer>"
        ]
        solutions = ["42"]
        weights = {"format": 0.2, "anomaly": 0.3, "correctness": 0.5}
        rewards = compute_rewards(completions, solutions, weights=weights)

        # format=1.0, anomaly=0.8 (text too short after strip), correctness=1.0
        # total = 0.2*1.0 + 0.3*0.8 + 0.5*1.0 = 0.94
        assert abs(rewards[0] - 0.94) < 0.01, f"Expected ~0.94, got {rewards[0]}"

    def test_combined_partial(self):
        completions = ["Just 42"]  # 无 format, 无 answer tag
        solutions = ["42"]
        rewards = compute_rewards(completions, solutions)

        # format=0.0, anomaly≈? (depends on text), correctness=0.0
        assert 0.0 <= rewards[0] <= 1.0

    def test_length_match(self):
        """输入列表长度必须匹配"""
        completions = ["a", "b"]
        solutions = ["a"]
        try:
            correctness_reward(completions, solutions)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # expected


# ============================================================
# 运行测试
# ============================================================
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])