import pandas as pd
import logging


def get_question(QA_path):
    """
    从Excel中读取每一行的问题，并通过 yield 每次返回一条问题
    :param QA_path: Excel 文件路径
    :return: 生成器，每次产出一个问题
    """
    try:
        df = pd.read_excel(QA_path)

        for idx, row in df.iterrows():
            question = row.get("问").strip()

            if not isinstance(question, str):
                logging.warning(f"跳过非字符串问题行: {question}")
                continue

            logging.info(f"读取问题: {question}")
            yield question

        logging.info("所有问题处理完毕")

    except Exception as e:
        print(f"读取文件时出错: {e}")


if __name__ == '__main__':
    excel_path = '/Users/ethanliu/Documents/LLM项目/BidMaster/resources/datas/QA_docs/QA_with_legal_basis.xlsx'
    _question = get_question(excel_path)
    for q in _question:
        print(q)
        break
    for q in _question:
        print(q)
        break
