# NOTE: knowledge_categories, knowledge_path和save_db_path必须一一对应（不对应则需要修改代码）
knowledge_categories = {"法律法规相关": 'laws', "基本知识相关": 'bid_basic_theory', "现有案例相关": 'bid_cases'}
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
llm_model = "Qwen/Qwen3-0.6B"
# emb_model = 'moka-ai/m3e-base'
emb_model = 'Qwen/Qwen3-0.6B'

# 句子切分配置
per_text_length = 50
cover_length = 15


# log file
log_file = '../logs/store_vectors_logs.log'

if __name__ == '__main__':
    pass

