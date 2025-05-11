# NOTE: knowledge_categories, knowledge_path和save_db_path必须一一对应（不对应则需要修改代码）
import torch

knowledge_categories = {"招投标法律规范相关": 'laws', "招投标流程相关": 'bid_basic_theory',
                        "现有招投标案例": 'bid_cases'}
# knowledge and save db path
pair_data_path = [
    {
        'laws_knowledge_path': '../knowledge/raw_knowledge/laws/',
        'save_laws_db_path': '../knowledge/vector_knowledge/laws.db',
    },
    {
        'bid_basic_theory_knowledge_path': '../knowledge/raw_knowledge/bid_basic_theory/',
        'save_bid_basic_theory_db_path': '../knowledge/vector_knowledge/bid_basic_theory.db',
    },
    {
        'bid_cases_knowledge_path': '../knowledge/raw_knowledge/bid_cases/',
        'save_bid_cases_db_path': '../knowledge/vector_knowledge/bid_cases.db'
    }

]

# 模型
llm_model = "../models/Qwen3-0.6B"
# llm_model = "Qwen/Qwen3-1.7B"
# llm_model = "Qwen/Qwen3-0.6B"

emb_model = '../models/Qwen3-0.6B'
# emb_model = 'moka-ai/m3e-base'
# emb_model = 'Qwen/Qwen3-0.6B'

# 句子切分配置
per_text_length = 512
cover_length = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# log file
# log_file = '../logs/store_vectors_logs.log'
# log_file = '../logs/chat_logs.log'
log_file = '../logs/total_logs.log'

# limit: for search tops limit
limit = 8

if __name__ == '__main__':
    pass
