# NOTE: knowledge_categories, knowledge_path和save_db_path必须一一对应（不对应则需要修改代码）
# NOTE: 另外knowledge_categories的对应的db也要对应正确
import torch
from utils.project_path import find_project_root

project_root = find_project_root()
knowledge_path = project_root / 'knowledge'
knowledge_categories: dict[str, str] = {"招投标法律规范相关": 'laws', "招投标流程相关": 'bid_basic_theory',
                                        "现有招投标案例": 'bid_cases'}

knowledge_categories_db: dict[str, str] = {'laws': knowledge_path / 'vector_knowledge/laws.db',
                                           'bid_basic_theory': knowledge_path / 'vector_knowledge/bid_basic_theory.db',
                                           'bid_cases': knowledge_path / 'vector_knowledge/bid_cases.db'}

# knowledge and save db path
pair_data_path: list[dict[str, str]] = [
    {
        'laws_knowledge_path': knowledge_path / 'raw_knowledge/laws/',
        'save_laws_db_path': knowledge_categories_db['laws'],
    },
    {
        'bid_basic_theory_knowledge_path': knowledge_path / 'raw_knowledge/bid_basic_theory/',
        'save_bid_basic_theory_db_path': knowledge_categories_db['bid_basic_theory'],
    },
    {
        'bid_cases_knowledge_path': knowledge_path / 'raw_knowledge/bid_cases/',
        'save_bid_cases_db_path': knowledge_categories_db['bid_cases']
    }
]

# 总的文档（不采取分类）
total_knowledge = {
    'total_name': '总文档',
    'total_collection_name': 'total_knowledge',
    'total_docs_path': knowledge_path / 'raw_knowledge/total_docs',
    'total_docs_db_path': knowledge_path / 'vector_knowledge/total_docs.db'
}

# 模型
model_path = project_root / 'resources/models/'
model_name = "qwen/Qwen3-0.6B"
llm_model = model_path / model_name
# llm_model = model_path / "qwen/Qwen3-1.7B"

# emb_model = model_path / 'moka-ai/m3e-base'
emb_model = model_path / "qwen/Qwen3-0.6B"

# 句子切分配置
per_text_length = 512
cover_length = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# log file
# log_file = project_root / 'logs/store_vectors_logs.log'
# log_file = project_root / 'logs/chat_logs.log'
# log_file = project_root / 'logs/total_logs.log'
log_file = project_root / 'logs/model_selection.log'

# limit: for search tops limit
limit = 10

if __name__ == '__main__':
    pass
