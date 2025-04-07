import os
import json

def create_jsonl(matching_file, formulas_file, images_dir, output_file):
    """
    将匹配文件和公式文件转换为 JSONL 格式，使用相对图片路径。

    目标格式示例 (适用于单张图片输入):
    {
        "messages": [
            {"role": "user", "content": "用户指令文本"},
            {"role": "assistant", "content": "助手回答文本"}
        ],
        "images": ["相对/图片/路径.jpg"] # 使用相对于JSONL文件的路径
    }

    Args:
        matching_file (str): 匹配文件的路径 (e.g., train.matching.txt)。
        formulas_file (str): 公式文件的路径 (e.g., train.formulas.norm.txt)。
        images_dir (str): 图片目录的绝对路径。
        output_file (str): 输出 JSONL 文件的绝对路径。
    """
    image_formula_map = {}
    try:
        with open(matching_file, 'r', encoding='utf-8') as f_match:
            for line_number, line in enumerate(f_match, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    image_name = parts[0].strip()
                    formula_id = parts[1].strip()
                    image_formula_map[image_name] = formula_id
                else:
                    print(f"警告: 匹配文件 '{matching_file}' 第 {line_number} 行格式错误: '{line}'. 跳过。")
    except FileNotFoundError:
        print(f"错误: 匹配文件未找到 '{matching_file}'")
        return

    formula_latex_map = {}
    try:
        with open(formulas_file, 'r', encoding='utf-8') as f_formula:
            for i, line in enumerate(f_formula):
                formula_latex_map[str(i)] = line.strip()
    except FileNotFoundError:
        print(f"错误: 公式文件未找到 '{formulas_file}'")
        return

    count_written = 0
    count_skipped_missing_formula = 0
    count_skipped_missing_image = 0

    # 获取输出JSONL文件的目录，作为计算相对路径的起点
    jsonl_dir = os.path.dirname(output_file)
    os.makedirs(jsonl_dir, exist_ok=True) # 确保目录存在

    with open(output_file, 'w', encoding='utf-8') as f_jsonl:
        for image_name, formula_id in image_formula_map.items():
            if formula_id in formula_latex_map:
                latex_formula = formula_latex_map[formula_id]
                # 1. 构建图片的绝对路径 (主要用于检查文件是否存在)
                absolute_image_path = os.path.join(images_dir, image_name)

                # 2. 检查图片文件是否存在 (使用绝对路径检查)
                if not os.path.exists(absolute_image_path):
                    # print(f"警告: 图片文件不存在 '{absolute_image_path}'. 跳过。")
                    count_skipped_missing_image += 1
                    continue

                # 3. 计算图片相对于JSONL文件目录的路径
                try:
                    # 使用 os.path.normpath 规范化路径以处理 '..' 等情况并确保斜杠正确
                    relative_image_path = os.path.relpath(absolute_image_path, jsonl_dir)
                    # 统一使用正斜杠 '/' 作为路径分隔符，提高跨平台兼容性
                    relative_image_path = relative_image_path.replace(os.sep, '/') 
                except ValueError as e:
                     # 如果文件在不同的驱动器上（Windows），relpath 会失败
                     print(f"错误: 无法计算相对路径从 '{jsonl_dir}' 到 '{absolute_image_path}'. 错误: {e}. 使用绝对路径作为后备。")
                     relative_image_path = absolute_image_path.replace(os.sep, '/') # 仍然统一斜杠


                # 构建符合目标格式的JSON对象
                data = {
                    "messages": [
                        {
                            "role": "user",
                            "content": "<image>What is the LaTeX formula of the math equation in this image"
                        },
                        {
                            "role": "assistant",
                            "content": latex_formula
                        }
                    ],
                    # 使用计算出的相对路径
                    "images": [relative_image_path]
                }
                f_jsonl.write(json.dumps(data, ensure_ascii=False) + '\n')
                count_written += 1
            else:
                # print(f"警告: 公式 ID '{formula_id}' 未找到。跳过。")
                count_skipped_missing_formula += 1

    print(f"处理完成: {output_file}")
    print(f"  成功写入条目: {count_written}")
    if count_skipped_missing_formula > 0:
        print(f"  因缺少公式ID而跳过的条目: {count_skipped_missing_formula}")
    if count_skipped_missing_image > 0:
         print(f"  因缺少图片文件而跳过的条目: {count_skipped_missing_image}")


# 主程序部分保持不变
if __name__ == "__main__":
    # 数据集基础路径
    base_dir = r"E:\dataset\dataset\Ovis_finetune\data\raw\fullhand"
    outbase_dir = r"E:\dataset\dataset\Ovis_finetune\data\processed\fullhand"
    # 定义各个子目录路径 (这些仍然是绝对路径，用于查找源文件)
    formulas_dir = os.path.join(base_dir, "formulas")
    images_dir = os.path.join(base_dir, "images")
    matching_dir = os.path.join(base_dir, "matching")

    print("开始处理训练集...")
    train_matching_file = os.path.join(matching_dir, "train.matching.txt")
    train_formulas_file = os.path.join(formulas_dir, "formulas.norm.txt")
    # 输出JSONL文件的绝对路径
    train_output_file = os.path.join(outbase_dir, "train.jsonl")
    # 注意：images_dir 传入的是绝对路径，函数内部会计算相对路径
    create_jsonl(train_matching_file, train_formulas_file, images_dir, train_output_file)

    print("\n开始处理验证集...")
    val_matching_file = os.path.join(matching_dir, "val.matching.txt")
    val_formulas_file = os.path.join(formulas_dir, "formulas.norm.txt")
    val_output_file = os.path.join(outbase_dir, "val.jsonl")
    create_jsonl(val_matching_file, val_formulas_file, images_dir, val_output_file)

    print("\n开始处理测试集...")
    test_matching_file = os.path.join(matching_dir, "test.matching.txt")
    test_formulas_file = os.path.join(formulas_dir, "formulas.norm.txt")
    test_output_file = os.path.join(outbase_dir, "test.jsonl")
    create_jsonl(test_matching_file, test_formulas_file, images_dir, test_output_file)

    print("\n数据集转换完成！")