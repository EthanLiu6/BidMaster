#  初步问答搭建
#  参考：https://python.langchain.com/docs/tutorials/llm_chain/
#       https://python.langchain.com/api_reference/deepseek/chat_models/langchain_deepseek.chat_models.ChatDeepSeek.html#langchain_deepseek.chat_models.ChatDeepSeek
#       https://api-docs.deepseek.com/zh-cn/
#       https://chat.langchain.com/?_gl=1*1ubhilj*_ga*MjAxNjg4NTE4Ni4xNzQ1NDA3NTU4*_ga_47WX3HKKY2*MTc0NTg2MTgxOC40LjAuMTc0NTg2MTgxOC4wLjAuMA..&threadId=793e20ca-a3f2-4913-ba1a-d6f4eaf567cc

from langchain.chat_models.base import init_chat_model
from langchain_deepseek import ChatDeepSeek

from docs_read import read_docx


def deepseek_model(model, model_provider, api_key):
    init_mode = init_chat_model(model, model_provider=model_provider, api_key=api_key)  # api_key是一种openAI接口规范
    return init_mode


if __name__ == '__main__':
    # deepseek
    model = 'deepseek-chat'  # deepseek-v3
    model_provider = 'deepseek'
    api_key = 'xxxxx'
    ds_model = deepseek_model(model, model_provider, api_key)


    file_path = r'../aihui_data/unstructural_data/laws/中华人民共和国招标投标法实施条例.docx'  # 替换成你的文件路径
    content = read_docx(file_path)
    count = 0
    for i in range(1, len(content), 512):
        print(content[i: i + 512])
        prompt = '你作为一个招投标法律助手，回答一下问题：\n'
        # print(ds_model.invoke(input=prompt + content[i: i + 512]))
        print(ctx for ctx in ds_model.stream(input=prompt + content[i: i + 512]))
        count += 1
        if count > 5:
            break

