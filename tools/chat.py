import logging
import os
import warnings
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from pymilvus.milvus_client import MilvusClient


from scripts import config
from scripts.config import knowledge_categories

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')


class QueryTool:
    """用户提问的底层LLM工具"""

    def __init__(self, _chat_model, valid_categories):
        self.tokenizer = _chat_model.tokenizer
        self.model = _chat_model.model
        self.valid_categories = valid_categories
        self.history = []

        logging.info(f'当前知识数据库含有类别：{self.valid_categories}')

    def classify_with_local_llm(self, user_query):
        try:
            # 构建提示
            prompt = f"""你是分类助手，请将用户的问题分类为以下类别之一：{self.valid_categories}。直接输出分类结果，不要解释。
    用户问题：{user_query}
    分类结果："""

            # 生成回答（使用贪婪解码）
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=16,
                # do_sample=False,  # 使用贪婪解码
                # temperature=0.01,  # 这个参数在贪婪解码下无效但无害
                # top_p=1.0,  # 设置为1.0表示不使用top-p采样
                # top_k=0.0  # 设置为0表示不使用top-k采样
            )

            # 解码并提取分类结果
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logging.info(f'当前知识类别分类回复：\n {response}')
            lines = response.split('\n')
            category = None
            for line in lines:
                print(line)
                if '分类结果：' in line:
                    category = line.split("分类结果：")[-1].strip()
                    break
            logging.info(f'当前识别类别：{category}')

            # eg: valid_categories = ["法律法规相关", "招投标基本知识", "招投标已有案例"]
            for valid in self.valid_categories:
                if valid in category:
                    return valid
            return "未知"

        except Exception as e:
            print("分类出错:", e)
            return "未知"

    # def generate_response(self, user_input):
    #     messages = self.history + [{"role": "user", "content": user_input}]
    #
    #     text = self.tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=False,
    #         add_generation_prompt=True
    #     )
    #
    #     inputs = self.tokenizer(text, return_tensors="pt").to(config.device)
    #     response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
    #     response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
    #
    #     # Update history
    #     self.history.append({"role": "user", "content": user_input})
    #     self.history.append({"role": "assistant", "content": response})
    #
    #     return response
    def generate_response(self, user_input, stream=False):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        if stream:
            # 流式生成
            from transformers import TextStreamer
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32768,
                streamer=streamer
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # 非流式生成（原逻辑）
            outputs = self.model.generate(**inputs, max_new_tokens=32768)
            response = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        # 更新历史记录
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response if not stream else None  # 流式模式下直接通过streamer输出，返回None


class Chat:
    """负责主要QA的chat"""

    def __init__(self, _chat_model):
        self.query_tool = QueryTool(_chat_model, valid_categories=knowledge_categories.keys())

    def chat_client(self):
        """符合openai标准规范的QA模型client"""
        pass

    def query_classify(self, user_query):
        return self.query_tool.classify_with_local_llm(user_query)

    @staticmethod
    def prompt_setting(context,
                       user_query,
                       system_prompt=None,
                       user_prompt=None,
                       ):
        if not system_prompt:
            SYSTEM_PROMPT = """
                Human: 你是一个招投标行业智能助手，你可以通过所给知识库内容找到问题的答案，并且能够自我整理内容并回复用户
                """
        else:
            SYSTEM_PROMPT = system_prompt
        if not user_prompt:
            USER_PROMPT = f"""
                使用以下包含在<context>标记中的信息片段来回答<question>标记中包含的问题。
                <context>
                {context}
                </context>
                <question>
                {user_query}
                </question>
                """
        else:
            USER_PROMPT = user_prompt

        return SYSTEM_PROMPT, USER_PROMPT

    def answer_generate(self, total_query, stream=True, *args):
        return self.query_tool.generate_response(user_input=total_query, stream=stream)


class ChatMilvusClient:
    def __init__(self, uri, *args):
        self._milvus_client = MilvusClient(uri=uri)

    def search_knowledge(self,
                         query,
                         _collection_name,
                         _chat_model,
                         limit=3
                         ) -> List[List[dict]]:
        """
        Returns:
                List[List[dict]]: A nested list of dicts containing the result data. Embeddings are
                not included in the result data.
        """
        question_emb = _chat_model.get_sentence_embedding(query)
        question_emb_list = question_emb.tolist()
        logging.info(f'当前问题进行embedding后的维度：{len(question_emb_list)}')

        _search_res = self._milvus_client.search(
            collection_name=_collection_name,
            data=[
                question_emb_list
                ],
            limit=limit,  # Return top limit results
            # TODO: metric_type改写到配置文件
            search_params={"metric_type": "COSINE", "params": {}},  # Inner product distance
            output_fields=["sentence", "from_doc"]
        )

        return _search_res


class ChatModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=config.device

        )
        self._encode_model = SentenceTransformer(model_name, device=config.device)


    def get_text2token_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        input_ids = inputs['input_ids']
        logging.info(f'当前text的token_ids：{input_ids}')
        embeddings = self.model.get_input_embeddings()(input_ids)
        logging.info(f'向量化后的 tensor shape：{embeddings.shape}')

        return embeddings

    def get_sentence_embedding(self, text):
        texts_embeddings = self._encode_model.encode(text)
        return texts_embeddings


if __name__ == '__main__':
    from tools import log_tool

    log_tool.set_log()

    # chat_model = ChatModel(model_name="../models/Qwen3-0.6B")
    chat_model = ChatModel(model_name="Qwen/Qwen3-0.6B")
    query_tool = QueryTool(_chat_model=chat_model,
                           valid_categories=list(knowledge_categories.keys())
                           )
    # query_tool = QueryTool(model_name="moka-ai/m3e-base")  # m3e做问答生成不行

    # question = "政府采购法对投标保证金有什么规定？"
    # question = "招投标的流程是怎样的"
    # question = "现有哪些采购的中标结果"
    question = "软件升级服务可以用单一来源采购方式吗？"
    cls = query_tool.classify_with_local_llm(question)
    print('*' * 30)
    print(cls)

    collection_name = knowledge_categories[cls]
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
    # full_response = ""
    # for chunk in answer_generate:
    #     print(chunk)
    #     if chunk.choices[0].delta.content:  # 检查是否有新内容
    #         content = chunk.choices[0].delta.content
    #         print(content, end="", flush=True)  # 逐字打印
    #         full_response += content




