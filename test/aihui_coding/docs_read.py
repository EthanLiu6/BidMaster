from docx import Document


def read_docx(file_path):

    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return ''.join(full_text)


if __name__ == '__main__':
    # 示例使用
    file_path = r'../aihui_data/unstructural_data/laws/中华人民共和国招标投标法实施条例.docx'
    content = read_docx(file_path)
    for i in range(1, len(content), 1024):
        print(content[i: i + 1024])
