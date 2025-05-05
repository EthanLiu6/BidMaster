import re
from pathlib import Path
from docx import Document
import pdfplumber


def read_docx(docx_path):
    doc = Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return ''.join(full_text)


# def clean_pdf_text(text):
#     """
#     专业法律条文清洗函数
#     处理特点：
#     1. 保留完整的条款编号（如"第三十四条"）
#     2. 规范化法律条文中的特殊格式（如"（一）"等）
#     3. 处理中文数字和阿拉伯数字混用情况
#     4. 智能分段和换行处理
#     """
#     # 1. 去除页眉页脚和网页信息
#     text = re.sub(r'\d{4}/\d{1,2}/\d{1,2} \d{1,2}:\d{2}', '', text)  # 去除日期时间
#     text = re.sub(r'第.*?国务院公报.*?政府网', '', text)
#     text = re.sub(r'\d+/\d+', '', text)  # 去除页码

#     # 2. 条款编号规范化处理
#     text = re.sub(r'第([一二三四五六七八九十百千]+)条', r'\n第\1条 ', text)

#     # 3. 处理中文数字和阿拉伯数字混用（如"第三十四"和"34"）
#     text = re.sub(r'第([0-9]+)条', lambda m: f'第{num2chinese(m.group(1))}条', text)

#     # 4. 法律条文特殊格式处理
#     text = re.sub(r'（([一二三四五六七八九十])）', r'（\1）', text)  # 统一括号格式
#     text = re.sub(r'([。；])\s*', r'\1\n', text)  # 句号和分号后换行

#     # 5. 处理列表项格式
#     text = re.sub(r'([（(][一二三四五六七八九十]+[）)])', r'\n\1', text)

#     # 6. 去除多余空行但保留段落分隔
#     text = re.sub(r'\n\s*\n', '\n\n', text)
#     text = re.sub(r'[ \t]+', ' ', text)  # 压缩空格

#     # 7. 章节标题处理
#     text = re.sub(r'第[一二三四五六七八九十]+章\s+.+', r'\n\g<0>\n', text)

#     # 8. 去网页垃圾字符
#     text = re.sub(r'http[s]\S+', '', text)
#     text = re.sub(r'©|®|™|•', '', text)

#     return text.strip()

def clean_pdf_text(text):
    """
    文本清洗
    """
    # 去除特殊字符但保留中文标点
    text = re.sub(r'[^\w\u4e00-\u9fff，。、；：？！「」『』（）《》【】\s]', '', text)
    # 规范化空白字符
    text = re.sub(r'\s+', ' ', text)
    # 去网页垃圾字符
    text = re.sub(r'http[s]\S+', '', text)
    text = re.sub(r'©|®|™|•', '', text)
    # 处理错误换行（英文单词）
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    # 处理错误换行（中文）
    text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)
    return text.strip()


def read_pdf(pdf_path):
    """
    基础文本提取
    :param pdf_path: PDF文件路径
    :return: 提取的文本字典 {page_num: text}
    """
    text_dict = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:  # 过滤空页
                text = clean_pdf_text(text)
                text_dict[i + 1] = text

    return text_dict
