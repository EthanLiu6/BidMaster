"""
仅仅是用于模型选型、模型评估测试
部分代码细节与实际之前的代码有所差异（比如generate_response）
"""

import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

from sentence_transformers import SentenceTransformer
from src import config
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TextIteratorStreamer
import threading

from question_get import get_question


class ChatModel:
    def __init__(self, QA_model_path, sentence_emb_model_name=None):
        self.QA_model_name = config.model_name
        self.QA_model_path = QA_model_path
        self.sentence_emb_model_name = sentence_emb_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.QA_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.QA_model_path,
            torch_dtype="auto",
            device_map=config.device
        )

        if not self.sentence_emb_model_name:
            # self.sentence_emb_model_name = self.QA_model_name
            self.sentence_encode_model = None
        else:
            self.sentence_encode_model = SentenceTransformer(self.sentence_emb_model_name, device=config.device)

    def get_text2token_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        input_ids = inputs['input_ids']
        logging.info(f'当前text的token_ids：{input_ids}')
        embeddings = self.model.get_input_embeddings()(input_ids)
        logging.info(f'向量化后的 tensor shape：{embeddings.shape}')

        return embeddings

    def get_sentence_embedding(self, text):
        texts_embeddings = self.sentence_encode_model.encode(text)
        return texts_embeddings


class ChatTool:
    def __init__(self, _chat_model: ChatModel):
        self.QA_model = _chat_model.model
        self.sentence_encode_model = _chat_model.sentence_encode_model
        self.tokenizer = _chat_model.tokenizer
        self.history = []

    @classmethod
    def set_prompt(cls, prompt: str = None):
        if not prompt:
            prompt = f"""你是一名招投标专业领域的智能助手，请根据问题进行相关回答。（不要进行深度思考）\n"""
        else:
            prompt = prompt
        return prompt

    def generate_response(self, user_input, is_set_prompt: bool = False, use_history=False, stream=False):
        if not is_set_prompt:
            if use_history:
                messages = self.history + [{"role": "user", "content": user_input}]
            else:
                messages = [{"role": "user", "content": user_input}]
        else:
            prompt = self.set_prompt()
            if use_history:
                messages = self.history + [{"role": "user", "content": prompt + user_input}]
            else:
                messages = [{"role": "user", "content": prompt + user_input}]

        if use_history:
            logging.info(f'当前对话轮数：{len(messages)}')

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=is_set_prompt,
            enable_thinking=config.enable_thinking
        )

        logging.info(f'apply_chat_template之后的内容长度：{len(text)}')

        inputs = self.tokenizer(text, return_tensors="pt").to(self.QA_model.device)

        if stream:
            # 流式打印生成并拿到回复结果
            streamer = CollectingStreamer(self.tokenizer)
            streamer.stream_generate(self.QA_model, inputs)
            _response = streamer.output

        else:
            # 非流式生成（原逻辑）
            outputs = self.QA_model.generate(**inputs, max_new_tokens=32768)
            _response = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        # 更新历史记录
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": _response})

        return _response


class Evaluation:
    def __init__(self, _chat_model: ChatModel, QA_store_path):
        self.QA_model_name = config.model_name
        self.chat_model = _chat_model
        self.QA_store_path = Path(QA_store_path)
        logging.info(f'当前使用模型：{self.QA_model_name}')

    def QA_without_prompt(self):
        pass

    def QA_with_prompt(self):
        pass

    def similarity(self, generate_text, label_text):
        generate_tokens = chat_model.tokenizer(generate_text)
        label_tokens = chat_model.tokenizer(label_text)
        print('生成文本与标签文本的tokens形状：')
        print(generate_tokens.shape)
        print(label_tokens.shape)

    def story_QA(self, user_question, generate_text):
        """保存问答情况到 Excel"""
        logging.info(f'当前用户问题：{user_question}')
        row = {
            '问': user_question,
            '答': generate_text,
            '模型': self.QA_model_name,
            '时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        if self.QA_store_path.exists():
            df = pd.read_excel(self.QA_store_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])

        df.to_excel(self.QA_store_path, index=False)
        logging.info(f"✅ 问答已保存至 {self.QA_store_path}")


class CollectingStreamer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.output = ""

    def stream_generate(self, model, inputs, max_new_tokens=4096):
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True
        )
        thread = threading.Thread(
            target=model.generate,
            kwargs={**inputs, "max_new_tokens": max_new_tokens, "streamer": streamer}
        )
        thread.start()

        for new_text in streamer:
            print(new_text, end="", flush=True)
            self.output += new_text


if __name__ == '__main__':

    from tools.log_tool import set_log

    set_log()

    chat_model = ChatModel(QA_model_path=config.llm_model)
    chat_tool = ChatTool(_chat_model=chat_model)
    evaluation = Evaluation(_chat_model=chat_model, QA_store_path=config.project_root + '/logs/QA_store.xlsx')

    set_prompt = False

    QA_path = config.project_root + '/resources/datas/QA_docs/QA_with_legal_basis.xlsx'
    for question in get_question(QA_path):
        if question.lower() == '':
            print("退出问答系统")
            break

        response = chat_tool.generate_response(user_input=question,
                                               is_set_prompt=set_prompt,
                                               use_history=False,
                                               stream=True)
        print('\n回复：\n', response)

        evaluation.story_QA(user_question=question, generate_text=response)
