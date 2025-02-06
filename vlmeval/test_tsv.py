from smp.vlm import encode_image_file_to_base64,decode_base64_to_image_file
import re
import json
import os
import csv
from tqdm import tqdm
from smp import *
import pandas as pd
# import uuid
# image_path = f"./{uuid.uuid4()}.jpg"
results=[]
header=["index","image","question","answer"]
from vlmeval.dataset.Omnidocbench.omnidocbench import *


# prompt = r'''You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

#         1. Text Processing:
#         - Accurately recognize all text content in the PDF image without guessing or inferring.
#         - Convert the recognized text into Markdown format.
#         - Maintain the original document structure, including headings, paragraphs, lists, etc.

#         2. Mathematical Formula Processing:
#         - Convert all mathematical formulas to LaTeX format.
#         - Enclose inline formulas with \( \). For example: This is an inline formula \( E = mc^2 \)
#         - Enclose block formulas with \\[ \\]. For example: \[ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]

#         3. Table Processing:
#         - Convert tables to HTML format.
#         - Wrap the entire table with <table> and </table>.

#         4. Figure Handling:
#         - Ignore figures content in the PDF image. Do not attempt to describe or convert images.

#         5. Output Format:
#         - Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
#         - For complex layouts, try to maintain the original document's structure and format as closely as possible.

#         Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
#         '''

#    python /mnt/petrelfs/wangfangdong/wang/VLMEvalKit/vlmeval/test_tsv.py
"""
gt_path='/mnt/petrelfs/wangfangdong/wang/OmniDocBench/OmniDocBench/OmniDocBench.json'
image_folder=os.path.dirname(gt_path)
image_folder_path=os.path.join(image_folder,'images')
# print(image_folder_path) /mnt/petrelfs/wangfangdong/wang/OmniDocBench/OmniDocBench/images

with open(gt_path,'r',encoding='utf-8') as f:
    gt_samples=json.load(f)


try:
    for i, sample in enumerate(tqdm(gt_samples, desc="Processing Samples")):
        image_name=sample["page_info"]["image_path"]
        image_abs_path=os.path.join(image_folder_path,image_name)
        if  not os.path.exists(image_abs_path):
            raise FileNotFoundError(f"Image not found: {image_abs_path}")
        index=i
        image_base64=encode_image_file_to_base64(image_abs_path)
        question=""
        # sample dict
        answer=json.dumps(sample)

        results.append([
            i,                 
            image_base64,         
            question.replace('\t', ' '),  
            answer
        ])
        
except Exception as e:
    print(f"\n处理中断，错误发生在第 {i+1} 个样本")
    print(f"错误信息: {str(e)}")
    raise

output_path = './omnidocbench.tsv'


with open(output_path,'w',newline='',encoding='utf-8') as f:
    writer = csv.writer(f,delimiter='\t')
    writer.writerow(header)
    writer.writerows(results)

print(f"\n处理完成，共处理 {len(results)} 个样本")
print(f"结果已保存至: {output_path}")
"""

eval_file='/mnt/petrelfs/wangfangdong/wang/VLMEvalKit/outputs/Qwen2-VL-2B-Instruct/Qwen2-VL-2B-Instruct_OmniDocBench.xlsx'

df = pd.read_excel(eval_file)
references = df['answer'].tolist()

# str->dict
for i,ans in enumerate(references):
    # # print(f'ans:{ans}')
    # print(f'{i}个数据正常')
    # # print(f'ans_type:{type(ans)}')
    # ans=json.loads(ans)
    # # print(f'ans_type:{type(ans)}')
    # self.references.append(ans) #[{},{}]

    try:
        ans = json.loads(ans)
        print(f'{i}个数据正常')
        # self.references.append(ans) #[{},{}]
    except json.JSONDecodeError as e:
        print(f"{i}个数据JSON 解析错误,{e}")
        continue
        
       




# index: 一个整数，tsv 中每一行的唯一标识
# image: 图片的 base64 编码，你可以使用 vlmeval/smp/vlm.py 中实现的API进行编码和解码：
# 编码：encode_image_to_base64（对于PIL Image）/ encode_image_file_to_base64（对于图片文件路径）
# 解码：decode_base64_to_image（对于PIL Image）/ decode_base64_to_image_file（对于图片文件路径）
# question: 针对图像所提取出的问题，类型为字符串
# answer: 问题的答案，图片所对应的真实gt


# data=load(data_path)
# # print(type(data)) <class 'pandas.core.frame.DataFrame'>
# for answer in data['answer']:
#     print(type(answer))
#     # str->dict
#     answer=json.loads(answer)
#     print(type(answer))
#     keys=answer.keys()
#     print(answer.keys())
#     print(type(answer['page_info']))
#     print(answer['page_info'])
  
#     print(answer['page_info']['image_path'])
#     break
# with open(gt_path,'r',encoding='utf-8') as f:
#     data=json.load(f)
# print(type(data))
# print(type(data[0]))

# answer=load(eval_file)['answer'].tolist()
# answer_dict=[]
# for ans in answer:
#     ans=json.loads(ans)
#     answer_dict.append(ans)
# print(type(answer))
# print(type(answer_dict[0]))
# print(answer_dict[0]==data[0])

# # 示例用法
# file_path = '/mnt/petrelfs/wangfangdong/wang/VLMEvalKit/omnidocbench.tsv'
# md5_value = calculate_md5(file_path)
# print(f"文件 {file_path} 的 MD5 值为: {md5_value}")
# eval_file ='/mnt/petrelfs/wangfangdong/wang/VLMEvalKit/omnidocbench.tsv'
# eval_file='/mnt/petrelfs/wangfangdong/wang/VLMEvalKit/outputs/Qwen2-VL-2B-Instruct/T20250204_Ga897a2c3/Qwen2-VL-2B-Instruct_OmniDocBench.xlsx'
# references = load(eval_file)['answer'].tolist()
# print(f'type_references:{type(references)}')
# print(f'len_references:{len(references)}')
# page_info={}
# for reference in references:
#     print(f'type_reference:{type(reference)}')
#     ans=json.loads(reference)
#     print(f'ans_type:{type(ans)}')
#     img_path=os.path.basename(ans['page_info']['image_path'])
#     print(f'img_path:{img_path}')
#     page_info[img_path]=ans['page_info']['page_attribute']
#     print(page_info)
#     break
# eval_file='/mnt/petrelfs/wangfangdong/wang/VLMEvalKit/outputs/Qwen2-VL-2B-Instruct/T20250205_Ga897a2c3/Qwen2-VL-2B-Instruct_OmniDocBench.xlsx'
# eval=Omnidocbenchend2endEvaluator(eval_file)
# metrics=eval.score()
# print(metrics)