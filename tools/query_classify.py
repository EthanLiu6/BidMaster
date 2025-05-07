from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings('ignore')


class QueryTool:
    """用户提问的底层LLM工具"""
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history = []

    def classify_with_local_llm(self, user_query):
        try:
            # 构建提示
            prompt = f"""你是分类助手，请将用户的问题分类为以下三类之一：法律法规相关、招投标基本知识、招投标已有案例。直接输出分类结果，不要解释。

    用户问题：{user_query}
    分类结果："""

            # 生成回答（使用贪婪解码）
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                # do_sample=False,  # 使用贪婪解码
                # temperature=0.01,  # 这个参数在贪婪解码下无效但无害
                # top_p=1.0,  # 设置为1.0表示不使用top-p采样
                # top_k=1.0  # 设置为0表示不使用top-k采样
            )

            # 解码并提取分类结果
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            category = response.split("分类结果：")[-1].strip()

            # 确保只返回三个类别中的一个
            # TODO: 类别设置为配置文件
            valid_categories = ["法律法规相关", "招投标基本知识", "招投标已有案例"]
            for valid in valid_categories:
                if valid in category:
                    return valid
            return "未知"

        except Exception as e:
            print("分类出错:", e)
            return "未知"


    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response



if __name__ == '__main__':

    query_tool = QueryTool(model_name="Qwen/Qwen3-0.6B")
    # query_tool = QueryTool(model_name="moka-ai/m3e-base")  # m3e做问答生成不行

    # question = "政府采购法对投标保证金有什么规定？"
    # question = "招投标的流程是怎样的"
    question = "现有哪些采购的中标结果"
    # question = "软件升级服务可以用单一来源采购方式吗？"
    print(query_tool.classify_with_local_llm(question))  # 应该输出"法律法规相关"


