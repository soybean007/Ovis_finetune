import os
from typing import List, Dict, Any
import editdistance 
import re

# 假设 ORM 是 Swift 提供的基类或 Swift 能识别的接口
# 如果 Swift 不提供明确的 ORM 基类，只需确保 __call__ 签名正确
try:
    # 尝试导入 Swift 可能提供的 ORM 基类（如果存在）
    from swift.llm.orm import ORMBase as ORM # 假设路径，请核对
except ImportError:
    # 如果没有基类，定义一个简单的占位符
    class ORM:
        def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
            raise NotImplementedError

# 假设 orms 是 Swift 在加载插件时会查找或提供的字典
# 如果不是全局字典，注册逻辑可能需要在 Swift 训练脚本中进行
orms: Dict[str, Any] = {} # 全局或特定上下文的 orms 字典

class LatexSimilarityORM(ORM):
    """
    基于 LaTeX 字符串相似度（使用字符错误率 CER）的奖励函数。
    """
    def calculate_cer(self, ground_truth, hypothesis):
        """计算字符错误率 (CER)。"""
        gt = ground_truth.strip()
        hyp = hypothesis.strip()
        if not gt:
            return 1.0 if hyp else 0.0
        dist = editdistance.eval(gt, hyp)
        return dist / len(gt)

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        """
        计算奖励 = 1 - CER。

        Args:
            completions (list[str]): 模型生成的 LaTeX 输出列表。
            solution (list[str]): 基准答案 LaTeX 字符串列表。

        Returns:
            list[float]: 奖励分数列表 (0 到 1 之间)。
        """
        rewards = []
        if len(completions) != len(solution):
             # 在实际应用中，可能需要更健壮的错误处理
             print(f"警告: completions 数量 ({len(completions)}) 与 solution 数量 ({len(solution)}) 不匹配。")
             # 返回一个合理的默认值或引发错误
             return [0.0] * len(completions) # 示例：为每个 completion 返回 0

        for completion, sol in zip(completions, solution):
            if not isinstance(completion, str): completion = str(completion) # 确保是字符串
            if not isinstance(sol, str): sol = str(sol) # 确保是字符串
                
            if not completion:
                reward = 1.0 if not sol else 0.0
            else:
                cer = self.calculate_cer(sol, completion)
                reward = max(0.0, 1.0 - cer) 
            rewards.append(reward)
        return rewards

# 注册自定义 ORM 到 orms 字典中
# Swift 在加载插件时会查找这个字典（或通过其他机制注册）
orms['latex_similarity_cer'] = LatexSimilarityORM() 
print("已注册自定义奖励模型: latex_similarity_cer") # 添加日志确认注册