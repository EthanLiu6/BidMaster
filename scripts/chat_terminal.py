"""
终端测试问答
"""
import logging

import config
from tools.chat import ChatModel, QueryTool, ChatMilvusClient, Chat


def chat_with_classify():
    chat_model = ChatModel(model_name=config.llm_model)
    query_tool = QueryTool(_chat_model=chat_model,
                           valid_categories=list(config.knowledge_categories.keys())
                           )

    while True:
        print("\n" + "=" * 50)
        question = input("请输入问题（输入 q 退出）: ").strip()

        # 退出条件
        if question.lower() == 'q':
            print("退出问答系统")
            break

        cls = query_tool.classify_with_local_llm(question)
        print('*' * 30)
        print(cls)

        if cls == '未知':
            chat_milvus_client = ChatMilvusClient(uri=config.total_knowledge['total_docs_db_path'])
            search_res = chat_milvus_client.search_knowledge(_chat_model=chat_model,
                                                             _collection_name=config.total_knowledge[
                                                                 'total_collection_name'],
                                                             limit=config.limit,
                                                             query=question)
        else:
            collection_name = config.knowledge_categories[cls]
            chat_milvus_client = ChatMilvusClient(uri=config.knowledge_categories_db[collection_name])
            search_res = chat_milvus_client.search_knowledge(_chat_model=chat_model,
                                                             _collection_name=collection_name,
                                                             limit=config.limit,
                                                             query=question)
        for res in search_res[0]:
            logging.info(f'查询到的匹配结果：{res}')
        context = "\n".join(
            [res['entity']['sentence'] for res in search_res[0]]
        )

        chat = Chat(_chat_model=chat_model)
        sys_prompt, user_prompt = chat.prompt_setting(context=context, user_query=question)
        answer_generate = chat.answer_generate(sys_prompt + user_prompt)
        print(answer_generate)


def chat_without_classify():
    """使用总文档直接查找并回答"""
    chat_model = ChatModel(model_name=config.llm_model)

    while True:
        print("\n" + "=" * 50)
        question = input("请输入问题（输入 q 退出）: ").strip()

        # 退出条件
        if question.lower() == 'q':
            print("退出问答系统")
            break

        chat_milvus_client = ChatMilvusClient(uri=config.total_knowledge['total_docs_db_path'])
        search_res = chat_milvus_client.search_knowledge(_chat_model=chat_model,
                                                         _collection_name=config.total_knowledge[
                                                             'total_collection_name'],
                                                         limit=config.limit,
                                                         query=question)
        for res in search_res[0]:
            logging.info(f'查询到的匹配结果：{res}')
        context = "\n".join(
            [res['entity']['sentence'] for res in search_res[0]]
        )

        chat = Chat(_chat_model=chat_model)
        sys_prompt, user_prompt = chat.prompt_setting(context=context, user_query=question)
        answer_generate = chat.answer_generate(sys_prompt + user_prompt)
        print(answer_generate)


if __name__ == '__main__':
    from tools import log_tool
    log_tool.set_log()
    choice = input("请选择问答内容方式(1为类别化QA，2为全局QA):\n").strip()
    if choice == '1':
        chat_with_classify()
    elif choice == '2':
        chat_without_classify()

