import os
import glob
import re

from docx import Document
import pdfplumber


__all__ = [
    'find_files_with_suffix',
    'read_file',
]


def find_files_with_suffix(directory, suffix):
    """
    用文件后缀名（例如：.txt）获取文件列表
    :param directory: 需要查找的文件夹
    :param suffix: 文件后缀名（例如：.pdf）
    :return: 获取到的文件路径列表
    """
    return glob.glob(os.path.join(directory, f"**/*{suffix}"), recursive=True)


def read_docx(docx_path):
    doc = Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return ''.join(full_text)


def clean_pdf_text(text):
    """
    pdf文本清洗
    """
    # 去除特殊字符但保留中文标点
    text = re.sub(r'[^\w\u4e00-\u9fff，。、；：？！「」『』（）《》【】\s]', '', text)
    # 规范化空白字符
    text = re.sub(r'\s+', ' ', text)
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


def read_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()  # 统一转为小写便于比较

    if ext == ".docx":
        return read_docx(file_path)
    elif ext == ".pdf":
        return read_pdf(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {ext}")


if __name__ == '__main__':
    # 示例用法
    directory_path = "../knowledge/raw_knowledge/laws/"
    suffix_name = ".docx"
    files_list = find_files_with_suffix(directory_path, suffix_name)
    for f in files_list:
        print(f)
