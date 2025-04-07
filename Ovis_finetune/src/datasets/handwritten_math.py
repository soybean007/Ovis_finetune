import os
import json
from typing import Dict, Any, List, Optional, Union

# 导入 Swift 相关模块 (具体路径可能依据您的安装和 Swift 版本有所不同)
# 通常需要导入 Dataset 和 register_dataset
# 以下为假设路径，请根据实际情况调整
try:
    from swift.llm import Dataset, register_dataset 
    from swift.utils.utils import get_logger
except ImportError:
    # 提供备用导入或提示，说明 Swift 库未正确安装或环境配置问题
    print("错误：无法导入 Swift 库模块。请确保 Swift 已正确安装并配置。")
    # 定义占位符以便代码能解析，但在实际运行时会失败
    def register_dataset(*args, **kwargs):
        def decorator(cls):
            return cls
        return decorator
    class Dataset:
        pass

logger = get_logger()

@register_dataset(
    dataset_name='handwritten-math-formula-jsonl', # 给您的数据集起一个唯一的名字
    task='multi-modal-chat', # 或适合您模型/任务的类型，例如 'image-to-text'
    # 其他元数据，例如描述
    description='Handwritten math formula recognition dataset from JSONL files.'
)
class HandwrittenMathDataset(Dataset):
    """
    从 JSONL 文件加载手写数学公式数据集。
    JSONL 格式:
    {
      "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "LATEX_GROUND_TRUTH"}
      ],
      "images": ["relative/path/image.png"]
    }
    
    预处理后输出格式 (供 GPRO 等使用):
    {
      'images': ['/absolute/path/image.png'], # 使用绝对路径
      'messages': [{'role': 'user', 'content': '...'}], # 仅用户消息
      'solution': 'LATEX_GROUND_TRUTH' # 单独的 solution 字段
    }
    """

    @classmethod
    def load_data(cls, dataset_path: str, **kwargs) -> List[Dict[str, Any]]:
        """
        加载并预处理 JSONL 数据集。
        Swift 会调用这个方法来获取数据。

        Args:
            dataset_path (str): 指向 JSONL 文件的路径。
                                Swift 的 dataset_mapping 通常会将 train/val/test 路径传入这里。
            **kwargs: 其他可能的参数 (例如 Swift 传递的 dataset_id 或 split 信息)

        Returns:
            List[Dict[str, Any]]: 预处理后的数据列表。
        """
        
        processed_data = []
        # 获取 JSONL 文件所在的目录，用于解析相对路径
        jsonl_dir = os.path.dirname(dataset_path)
        logger.info(f"开始加载数据集: {dataset_path}")
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"跳过行 {i+1}: JSON 解析错误 - {e}")
                        continue

                    if 'messages' not in row or 'images' not in row:
                        logger.warning(f"跳过行 {i+1}: 缺少 'messages' 或 'images' 字段。")
                        continue
                        
                    messages = row.get('messages', [])
                    relative_image_paths = row.get('images', [])

                    user_message = None
                    ground_truth_latex = None

                    if len(messages) >= 2 and messages[0]['role'] == 'user' and messages[1]['role'] == 'assistant':
                        user_message = messages[0]
                        ground_truth_latex = messages[1].get('content', '').strip()
                    else:
                        logger.warning(f"跳过行 {i+1}: 'messages' 格式不符合预期的 [user, assistant] 结构。")
                        continue

                    if not ground_truth_latex:
                        logger.warning(f"跳过行 {i+1}: 无法从助手消息中提取基准答案 LaTeX。")
                        continue
                        
                    # --- 将相对图像路径转换为绝对路径 ---
                    absolute_image_paths = []
                    valid_entry = True
                    for rel_path in relative_image_paths:
                        abs_path = os.path.normpath(os.path.join(jsonl_dir, rel_path))
                        if not os.path.exists(abs_path):
                            logger.warning(f"跳过行 {i+1}: 图片文件不存在 '{abs_path}' (由相对路径 '{rel_path}' 解析得到)。")
                            valid_entry = False
                            break # 如果一个图片缺失，则跳过整个条目
                        absolute_image_paths.append(abs_path)
                    
                    if not valid_entry:
                        continue # 跳到下一行

                    # --- 构建 Swift 训练器期望的格式 ---
                    processed_row = {
                        'images': absolute_image_paths, 
                        'messages': [user_message], 
                        'solution': ground_truth_latex 
                    }
                    processed_data.append(processed_row)

        except FileNotFoundError:
            logger.error(f"错误: 数据集文件未找到 '{dataset_path}'")
            # 可能需要引发更严重的错误或返回空列表
            return []
        except Exception as e:
            logger.error(f"加载数据集 '{dataset_path}' 时发生未知错误: {e}")
            return []
            
        logger.info(f"加载并预处理完成: {dataset_path}, 共 {len(processed_data)} 条有效数据。")
        return processed_data