import logging
import os
from tools.log_tool import set_log

from utils.text_utils import TextUtils
from utils import docs_utils
from tools.docs_tools import DocsTools
import config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

knowledge_categories = config.knowledge_categories
# 字典类型不同类别原数据和存储数据的路径
pair_data_path = config.pair_data_path
emb_model = config.emb_model
per_text_length = config.per_text_length
cover_length = config.cover_length

total_knowledge = config.total_knowledge


def store_with_classify():
    categories = list(knowledge_categories.values())
    logging.info(f"类别列表：{categories}（英文，用于向量库的'collection_name'）")
    for categories_idx, per_pair_data_path in enumerate(pair_data_path):
        category = categories[categories_idx]

        knowledge_and_db_path = list(per_pair_data_path.keys())
        logging.info(f"当前检索到的原数据路径与向量存储路径：{per_pair_data_path}")
        logging.info(f"当前检索到的原数据类别：{category}")

        docxs_path_list = DocsTools.get_docxs(per_pair_data_path[knowledge_and_db_path[0]])
        logging.info(f"读取到的'{category}'类别所有docx路径：{docxs_path_list}")
        pdfs_path_list = DocsTools.get_pdfs(per_pair_data_path[knowledge_and_db_path[0]])  # raw knowledge
        logging.info(f"读取到的'{category}'类别所有pdf路径：{pdfs_path_list}")

        save_db_path = per_pair_data_path[knowledge_and_db_path[1]]
        text_utils = TextUtils(save_db_path=save_db_path,
                               encode_model=emb_model,
                               collection_name=category)
        logging.info(f'创建向量数据库：{save_db_path}')
        logging.info(f'使用向量模型：{emb_model}')

        for docx_path in docxs_path_list:
            logging.info(f'读取的当前文件：{docx_path}')
            docx_texts = docs_utils.read_docx(docx_path=docx_path)
            docx_chunk_texts = text_utils.chunk_texts(origin_texts=docx_texts,
                                                      per_text_length=per_text_length,
                                                      cover_length=cover_length)
            logging.info(f'当前文档切分后的文本列表大小：{(len(docx_chunk_texts), len(docx_chunk_texts[0]))}')

            docx_embeddings, embeddings_dim = text_utils.text2vector(docx_chunk_texts)
            logging.info(f'切分后的文本列表向量化大小：{docx_embeddings.shape}')

            text_utils.insert_vector(chunk_texts_list=docx_chunk_texts,
                                     texts_embeddings=docx_embeddings,
                                     embedding_dim=embeddings_dim,
                                     from_doc=docx_path)

        for pdf_path in pdfs_path_list:
            logging.info(f'读取的当前文件：{pdf_path}')
            page_and_pdf_texts = docs_utils.read_pdf(pdf_path=pdf_path)
            pdf_texts = ''
            for page in page_and_pdf_texts:
                pdf_texts += page_and_pdf_texts[page]
            logging.info(f"当前pdf成功转为text，预进行切片")

            pdf_chunk_texts = text_utils.chunk_texts(origin_texts=pdf_texts,
                                                     per_text_length=per_text_length,
                                                     cover_length=cover_length)

            logging.info("切片成功，进行向量存储...")
            pdf_embeddings, embeddings_dim = text_utils.text2vector(pdf_chunk_texts)
            text_utils.insert_vector(chunk_texts_list=pdf_chunk_texts,
                                     texts_embeddings=pdf_embeddings,
                                     embedding_dim=embeddings_dim,
                                     from_doc=pdf_path)
            logging.info("当前pdf已向量化存储")


def store_without_classify():
    """使用全文档存储"""
    category = total_knowledge['total_name']

    total_docs_path = total_knowledge['total_docs_path']
    total_docs_db_path = total_knowledge['total_docs_db_path']
    total_collection_name = total_knowledge['total_collection_name']
    logging.info(f"当前检索到的原数据路径：{total_docs_path}\n 向量存储路径：{total_docs_db_path}")
    logging.info(f"当前检索到的原数据类别：{category}")

    docxs_path_list = DocsTools.get_docxs(total_docs_path)
    logging.info(f"读取到的'{category}'类别所有docx路径：{docxs_path_list}")
    pdfs_path_list = DocsTools.get_pdfs(total_docs_path)  # raw knowledge
    logging.info(f"读取到的'{category}'类别所有pdf路径：{pdfs_path_list}")

    text_utils = TextUtils(save_db_path=total_docs_db_path,
                           encode_model=emb_model,
                           collection_name=total_collection_name)
    logging.info(f'创建向量数据库：{total_docs_db_path}')
    logging.info(f'使用向量模型：{emb_model}')

    for docx_path in docxs_path_list:
        logging.info(f'读取的当前文件：{docx_path}')
        docx_texts = docs_utils.read_docx(docx_path=docx_path)
        docx_chunk_texts = text_utils.chunk_texts(origin_texts=docx_texts,
                                                  per_text_length=per_text_length,
                                                  cover_length=cover_length)
        logging.info(f'当前文档切分后的文本列表大小：{(len(docx_chunk_texts), len(docx_chunk_texts[0]))}')

        docx_embeddings, embeddings_dim = text_utils.text2vector(docx_chunk_texts)
        logging.info(f'切分后的文本列表向量化大小：{docx_embeddings.shape}')

        text_utils.insert_vector(chunk_texts_list=docx_chunk_texts,
                                 texts_embeddings=docx_embeddings,
                                 embedding_dim=embeddings_dim,
                                 from_doc=docx_path)

    for pdf_path in pdfs_path_list:
        logging.info(f'读取的当前文件：{pdf_path}')
        page_and_pdf_texts = docs_utils.read_pdf(pdf_path=pdf_path)
        pdf_texts = ''
        for page in page_and_pdf_texts:
            pdf_texts += page_and_pdf_texts[page]
        logging.info(f"当前pdf成功转为text，预进行切片")

        pdf_chunk_texts = text_utils.chunk_texts(origin_texts=pdf_texts,
                                                 per_text_length=per_text_length,
                                                 cover_length=cover_length)

        logging.info("切片成功，进行向量存储...")
        pdf_embeddings, embeddings_dim = text_utils.text2vector(pdf_chunk_texts)
        text_utils.insert_vector(chunk_texts_list=pdf_chunk_texts,
                                 texts_embeddings=pdf_embeddings,
                                 embedding_dim=embeddings_dim,
                                 from_doc=pdf_path,)
        logging.info("当前pdf已向量化存储")


if __name__ == '__main__':
    set_log()
    choice = input("请选择向量存储方式(1为类别化存储，2为总体存储):\n").strip()
    if choice == '1':
        store_with_classify()
    else:
        store_without_classify()

