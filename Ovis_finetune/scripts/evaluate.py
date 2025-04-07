# scripts/evaluate.py

import argparse
import json
import os
import datetime # 用于在摘要中添加时间戳
from PIL import Image
import torch
from tqdm import tqdm # 用于显示进度条
import editdistance # 用于计算编辑距离 (CER, WER)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction # 用于计算 BLEU
from typing import List, Dict, Any, Optional

# 尝试导入 Hugging Face 库
try:
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig # 或根据您的模型选择 AutoModelForVision2Seq
    from peft import PeftModel # 用于加载 LoRA 等适配器
except ImportError:
    print("错误：找不到必要的库 (torch, transformers, peft, accelerate)。")
    print("请使用以下命令安装: pip install torch transformers peft accelerate")
    exit(1)

# --- 评估指标计算函数 ---
# (在此处包含之前讨论过的 calculate_em, calculate_cer, calculate_wer, calculate_bleu, tokenize_latex 函数)

def tokenize_latex(latex_string: str) -> List[str]:
    """
    简单的 LaTeX 分词器（按空格分割）。
    如果需要，可以用更复杂的分词器替换。
    """
    # 可以考虑使用正则表达式进行更精确的分词:
    # import re
    # tokens = re.findall(r'\\(?:[a-zA-Z]+|[^\s\\])|[a-zA-Z0-9]+|\S', latex_string)
    return latex_string.strip().split()

def calculate_em(ground_truth_latex: str, model_output_latex: str) -> float:
    """计算完全匹配率 (Exact Match)。"""
    return 1.0 if ground_truth_latex.strip() == model_output_latex.strip() else 0.0

def calculate_cer(ground_truth_latex: str, model_output_latex: str) -> float:
    """计算字符错误率 (Character Error Rate)。"""
    gt = ground_truth_latex.strip()
    hyp = model_output_latex.strip()
    if not gt: # 如果基准答案为空
        return 1.0 if hyp else 0.0 # 如果模型输出也为空则错误率为0，否则为1
    # 计算 Levenshtein 距离
    dist = editdistance.eval(gt, hyp)
    # 返回错误率
    return dist / len(gt)

def calculate_wer(ground_truth_latex: str, model_output_latex: str) -> float:
    """使用 tokenize_latex 计算词错误率 (Word Error Rate)。"""
    gt_tokens = tokenize_latex(ground_truth_latex)
    hyp_tokens = tokenize_latex(model_output_latex)
    if not gt_tokens:
        return 1.0 if hyp_tokens else 0.0
    dist = editdistance.eval(gt_tokens, hyp_tokens)
    return dist / len(gt_tokens)

def calculate_bleu(ground_truth_latex: str, model_output_latex: str) -> float:
    """使用 tokenize_latex 计算 BLEU 分数。"""
    gt_tokens = tokenize_latex(ground_truth_latex)
    hyp_tokens = tokenize_latex(model_output_latex)
    reference = [gt_tokens] # 参考答案需要是列表的列表
    hypothesis = hyp_tokens
    # 使用平滑函数处理n-gram不匹配的情况
    smoothing_function = SmoothingFunction().method1 # 可以选择 method0-method7
    try:
        score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)
    except ZeroDivisionError: # 如果没有任何匹配且未使用平滑，可能会发生错误
        score = 0.0
    return score

# --- 主要评估逻辑 ---

def load_model_and_processor(model_path: str, adapter_path: Optional[str], use_bf16: bool, device: torch.device):
    """加载模型和处理器。"""
    # 根据可用性选择数据类型
    dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_bf16_supported() else torch.float16
    print(f"使用数据类型: {dtype}")

    print(f"从 {model_path} 加载处理器...")
    # 对于某些自定义模型/处理器，可能需要 trust_remote_code=True
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print(f"从 {model_path} 加载基础模型...")
    # 重要提示: 如果您的模型不是通用的 CausalLM（例如是 Vision2Seq 模型），请替换 AutoModelForCausalLM
    # 如果使用了量化（如 QLoRA），在此处添加 BitsAndBytesConfig
    # bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True
        # low_cpu_mem_usage=True, # 加载大模型时可能需要
        # quantization_config=bnb_config # 如果使用了量化
    )

    # 如果提供了适配器路径，则加载 PEFT 适配器 (例如 LoRA)
    if adapter_path:
        print(f"从 {adapter_path} 加载 PEFT 适配器...")
        try:
            model = PeftModel.from_pretrained(model, adapter_path)
            print("PEFT 适配器加载成功。")
            # 可选：如果显存允许，合并权重以可能加速推理
            # print("正在合并适配器权重...")
            # model = model.merge_and_unload()
            # print("权重合并完成。")
        except Exception as e:
            print(f"加载适配器时出错: {e}。将仅使用基础模型进行评估。")

    model.to(device) # 将模型移动到指定设备
    model.eval() # 设置为评估模式（禁用 dropout 等）
    print("模型和处理器加载完成。")
    return model, processor

def load_test_data(test_data_path: str, dataset_root: Optional[str]) -> List[Dict[str, Any]]:
    """从 JSONL 文件加载测试数据。"""
    print(f"从 {test_data_path} 加载测试数据...")
    test_samples = []
    # 获取 JSONL 文件所在的目录，用作解析相对路径的基准（如果 dataset_root 未提供）
    jsonl_dir = os.path.dirname(test_data_path) 

    try:
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    # 使用预处理脚本生成的结构
                    if 'messages' in data and 'solution' in data and 'images' in data and data['images']:
                        # 假设 'solution' 字段包含基准答案 LaTeX
                        solution = data['solution']
                        # 假设 'messages' 包含用户提示 (对于某些模型可能需要)
                        user_prompt = data['messages'][0]['content'] if data['messages'] else "" 
                        
                        # --- 解析图像路径 ---
                        image_rel_path = data['images'][0] # 假设每个条目只有一个图像
                        image_path = "" # 初始化
                        
                        if os.path.isabs(image_rel_path):
                             # 如果路径已经是绝对路径
                             image_path = image_rel_path
                        elif dataset_root:
                            # 如果提供了 dataset_root，则路径相对于它
                            image_path = os.path.join(dataset_root, image_rel_path)
                        else:
                             # 如果没有 dataset_root，则路径相对于 JSONL 文件本身
                             image_path = os.path.normpath(os.path.join(jsonl_dir, image_rel_path))
                        
                        # 检查解析后的图像文件是否存在
                        if os.path.exists(image_path):
                            test_samples.append({
                                'id': f"sample_{i}", # 添加一个唯一标识符
                                'image_path': image_path, # 使用解析后的路径
                                'prompt': user_prompt, 
                                'ground_truth': solution
                            })
                        else:
                            print(f"警告 [行 {i+1}]: 在解析后的路径 '{image_path}' 未找到图像 (源自 '{image_rel_path}')。跳过此样本。")
                    else:
                        print(f"警告 [行 {i+1}]: 跳过格式错误的行 (缺少字段): {line.strip()}")
                except json.JSONDecodeError:
                    print(f"警告 [行 {i+1}]: 跳过无效的 JSON 行: {line.strip()}")
    except FileNotFoundError:
        print(f"错误: 测试数据文件未找到于 '{test_data_path}'")
        return []
    except Exception as e:
        print(f"加载测试数据时发生错误: {e}")
        return []
        
    print(f"成功加载 {len(test_samples)} 个测试样本。")
    return test_samples

def run_evaluation(args):
    """执行完整的评估流程。"""
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and 'cpu' not in args.device else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型和处理器
    model, processor = load_model_and_processor(args.model_path, args.adapter_path, args.bf16, device)
    
    # 加载测试数据
    test_data = load_test_data(args.test_data_path, args.dataset_root)

    if not test_data:
        print("未加载有效的测试数据。正在退出。")
        return

    # 初始化用于存储结果的列表
    predictions = []
    ground_truths = []
    results_data = [] # 用于保存详细结果

    print(f"\n在 {len(test_data)} 个样本上运行推理...")
    # 注意：批处理 (batch_size > 1) 需要对循环进行重大修改以处理每步多个图像/提示。
    # 此代码目前假定 batch_size=1。
    if args.batch_size > 1:
         print("警告：此脚本的循环目前最适合 batch_size=1。批处理 > 1 需要修改。")

    # 遍历测试数据
    for sample in tqdm(test_data, desc="正在评估"):
        gt = sample['ground_truth']
        ground_truths.append(gt)
        image_path = sample['image_path']
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"\n加载图像时出错 {image_path}: {e}。跳过此样本。")
            predictions.append(f"错误_加载图像: {e}") # 添加错误占位符
            # 确保 metrics 也能处理这种情况或跳过
            continue

        # --- 准备模型输入 ---
        # !!! 关键步骤：如何格式化输入很大程度上取决于模型架构 !!!
        # 例如，一些模型（如 Qwen-VL, IDEFICS）可能需要特定的聊天模板和图像标记
        # prompt_text = f"图中的数学公式是什么？\n<image>" # 示例提示
        # inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(device)
        
        # 另一个示例，如果模型可以直接处理文本+图像
        # 使用数据中的原始提示或一个标准化的任务提示
        prompt_text = "识别图中数学公式的 LaTeX 表达式：" # 或使用 sample['prompt']
        try:
            inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(device)

            # 确保输入张量的数据类型与模型匹配
            if 'pixel_values' in inputs and hasattr(model, 'dtype') and inputs['pixel_values'].dtype != model.dtype:
                inputs['pixel_values'] = inputs['pixel_values'].to(model.dtype)
        except Exception as e:
            print(f"\n处理输入时出错 {image_path}: {e}。跳过此样本。")
            predictions.append(f"错误_处理输入: {e}")
            continue

        # --- 生成预测 ---
        generated_ids = None # 初始化
        try:
            with torch.no_grad(): # 推理时不需要计算梯度
                # 根据需要调整生成参数
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens, # 最大生成长度
                    num_beams=args.num_beams,           #束搜索宽度 (1 表示贪婪解码)
                    do_sample=False,                    # 评估时通常不采样
                    # 设置 pad_token_id 为 eos_token_id，如果 pad_token_id 未定义
                    pad_token_id=processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id, 
                    eos_token_id=processor.tokenizer.eos_token_id # 明确指定 EOS token ID
                )
        except Exception as e:
            print(f"\n模型生成时出错 {image_path}: {e}")
            predictions.append(f"错误_模型生成: {e}") # 添加错误占位符
            continue

        # --- 解码预测结果 ---
        # 需要从生成的 ID 中移除提示部分的 token
        # 注意：根据模型和 generate 设置，提示 token 可能包含在 generated_ids 中，也可能不包含
        pred = "解码失败" # 默认值
        try:
            input_token_len = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
            
            if generated_ids is not None:
                # 检查生成的是否比输入长
                if generated_ids.shape[1] > input_token_len:
                     # 假设 batch size 为 1，并切掉输入部分
                     output_ids = generated_ids[0, input_token_len:] 
                     pred = processor.decode(output_ids, skip_special_tokens=True).strip()
                else: # 可能只生成了 EOS 或没有生成新内容
                     # 尝试解码整个序列，可能需要后处理移除提示
                     full_decode = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
                     # 这里的后处理逻辑依赖于模型是否在输出中重复提示
                     print(f"警告: 模型没有为 {image_path} 生成新的 token (或只生成了 EOS)。完整解码: '{full_decode}'")
                     # 假设没有新内容生成，或者根据 full_decode 判断
                     pred = "" # 或进行更智能的提取

        except Exception as e:
             print(f"\n解码预测时出错 {image_path}: {e}")
             # pred 保持为 "解码失败" 或其他错误指示

        predictions.append(pred)

        # 计算此样本的指标
        em = calculate_em(gt, pred)
        cer = calculate_cer(gt, pred)
        wer = calculate_wer(gt, pred)
        bleu = calculate_bleu(gt, pred)

        # 存储详细结果以供后续分析或保存
        results_data.append({
            'id': sample['id'],
            'image_path': image_path, 
            'ground_truth': gt,
            'prediction': pred,
            'EM': em,
            'CER': cer,
            'WER': wer,
            'BLEU': bleu
        })

    # --- 汇总并报告指标 ---
    print("\n正在计算总体评估指标...")
    total_em = sum(item['EM'] for item in results_data)
    all_cer = [item['CER'] for item in results_data]
    all_wer = [item['WER'] for item in results_data]
    all_bleu = [item['BLEU'] for item in results_data]

    num_samples = len(results_data)
    # 计算平均值，处理空列表的情况
    avg_cer = sum(all_cer) / num_samples if num_samples > 0 else 0
    avg_wer = sum(all_wer) / num_samples if num_samples > 0 else 0
    avg_bleu = sum(all_bleu) / num_samples if num_samples > 0 else 0
    em_rate = total_em / num_samples if num_samples > 0 else 0

    # 打印结果
    print("\n--- 整体评估结果 ---")
    print(f"总评估样本数: {num_samples}")
    print(f"完全匹配率 (EM Rate): {em_rate:.4f}")
    print(f"平均字符错误率 (Avg CER): {avg_cer:.4f}")
    print(f"平均词错误率 (Avg WER): {avg_wer:.4f}")
    print(f"平均 BLEU 分数: {avg_bleu:.4f}")
    print("---------------------")

    # --- 保存详细结果 (可选) ---
    if args.output_dir:
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            # 保存每个样本的详细结果为 JSON Lines 文件
            output_results_file = os.path.join(args.output_dir, "evaluation_results_detail.jsonl")
            with open(output_results_file, 'w', encoding='utf-8') as f:
                for item in results_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"详细评估结果已保存至: {output_results_file}")

            # 保存评估摘要信息
            output_summary_file = os.path.join(args.output_dir, "evaluation_summary.json")
            summary = {
                "eval_timestamp": datetime.datetime.now().isoformat(), # 记录评估时间
                "model_path": args.model_path,
                "adapter_path": args.adapter_path,
                "test_data_path": args.test_data_path,
                "num_samples": num_samples,
                "metrics": {
                    "em_rate": em_rate,
                    "avg_cer": avg_cer,
                    "avg_wer": avg_wer,
                    "avg_bleu": avg_bleu,
                },
                "generation_config": { # 记录生成参数
                    "max_new_tokens": args.max_new_tokens,
                    "num_beams": args.num_beams,
                }
            }
            with open(output_summary_file, 'w', encoding='utf-8') as f:
                 json.dump(summary, f, indent=4, ensure_ascii=False)
            print(f"评估摘要已保存至: {output_summary_file}")
        except Exception as e:
            print(f"将结果保存到 {args.output_dir} 时出错: {e}")

# --- 命令行参数解析 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估用于数学公式识别的微调多模态模型")
    
    # 模型加载参数
    parser.add_argument("--model_path", type=str, required=True, help="基础模型目录路径 (Hugging Face 格式)。")
    parser.add_argument("--adapter_path", type=str, default=None, help="(可选) PEFT 适配器权重目录路径 (例如 LoRA 检查点)。")
    parser.add_argument("--processor_path", type=str, default=None, help="处理器/分词器目录路径。如果未提供，则默认为 model_path。")
    
    # 数据参数
    parser.add_argument("--test_data_path", type=str, required=True, help="测试集 JSONL 文件路径。")
    parser.add_argument("--dataset_root", type=str, default=None, help="用于解析 JSONL 中相对图像路径的数据集根目录。如果未提供，则假定路径是相对于 JSONL 文件或绝对路径。")
    
    # 推理参数
    parser.add_argument("--batch_size", type=int, default=1, help="推理时的批处理大小。注意：当前实现主要针对 batch_size=1 优化。")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="为 LaTeX 公式生成的最大新 token 数。")
    parser.add_argument("--num_beams", type=int, default=1, help="束搜索的宽度 (1 表示贪婪解码)。")
    
    # 执行参数
    parser.add_argument("--device", type=str, default="cuda:0", help="用于推理的设备 (例如 'cuda:0', 'cuda', 'cpu')。")
    parser.add_argument("--bf16", action="store_true", help="如果可用，使用 bfloat16 精度。")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default=None, help="(可选) 用于保存详细评估结果和摘要的目录。")
        
    args = parser.parse_args()

    # 如果未指定 processor_path，则使用 model_path
    if not args.processor_path:
        args.processor_path = args.model_path

    # 运行评估
    run_evaluation(args)
    
    print("评估脚本执行完毕。")