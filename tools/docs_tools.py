from utils.docs_utils import find_files_with_suffix

__all__ = ['DocsTools']


class DocsTools:
    def __init__(self):
        pass

    @staticmethod
    def get_pdfs(directory):  # pdf文档
        return find_files_with_suffix(directory=directory, suffix='.pdf')

    @staticmethod
    def get_docxs(directory):  # word文档
        return find_files_with_suffix(directory=directory, suffix='.docx')

    @staticmethod
    def get_txts(directory):  # txt
        return find_files_with_suffix(directory=directory, suffix='.txt')


if __name__ == '__main__':
    pass
