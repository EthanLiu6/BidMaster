from sentence_transformers import SentenceTransformer, util

# model = SentenceTransformer("moka-ai/m3e-base")
# model = SentenceTransformer("moka-ai/m3e-small")
# model = SentenceTransformer("moka-ai/m3e-large")

# model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

# model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
# model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

# model = SentenceTransformer('maidalun1020/bce-embedding-base_v1')

# model = SentenceTransformer('chestnutlzj/ChatLaw-Text2Vec')

# model = SentenceTransformer('richinfoai/ritrieve_zh_v1')

# model = SentenceTransformer('thenlper/gte-large-zh')
# model = SentenceTransformer('thenlper/gte-base-zh')
# model = SentenceTransformer('thenlper/gte-small-zh')

# model = SentenceTransformer('ByteDance-Seed/Seed1.5-Embedding')

querys = [
    "中标公告后，能否要求中标供应商自行来代理机构处领取中标通知书？",
    '专门面向中小企业的采购项目，监狱企业和残疾人福利性单位可以来投标吗？',
    '把赠予的款项写入招标文件可以吗？',
    '政府采购项目中，被质疑的主体是采购人还是供应商？',
    '招标文件中能否要求本项目除进口产品外，其他产品不得投标？'
]
sentences = [
    "《政府采购货物和服务招标投标管理办法》（财政部令第87号）第六十九条第五款 在公告中标结果的同时，采购人或者采购代理机构应当向中标人发出中标通知书；对未通过资格审查的投标人，应当告知其未通过的原因；采用综合评分法评审的，还应当告知未中标人本人的评审得分与排序。",
    "《关于促进残疾人就业政府采购政策的通知》（财库〔2017〕141号）：三、在政府采购活动中，残疾人福利性单位视同小型、微型企业，享受预留份额、评审中价格扣除等促进中小企业发展的政府采购政策。向残疾人福利性单位采购的金额，计入面向中小企业采购的统计数据。残疾人福利性单位属于小型、微型企业的，不重复享受政策。",
    "《政府采购法》第二条第四款 本法所称采购，是指以合同方式有偿取得货物、工程和服务的行为，包括购买、租赁、委托、雇用等。 《政府采购法实施条例》第十一条第二款 采购人不得向供应商索要或者接受其给予的赠品、回扣或者与采购无关的其他商品、服务。",
    '《政府采购质疑和投诉办法》（财政部令第94号）第十条第一款 供应商认为采购文件、采购过程、中标或者成交结果使自己的权益受到损害的，可以在知道或者应知其权益受到损害之日起7个工作日内，以书面形式向采购人、采购代理机构提出质疑。',
    '《财政部办公厅关于政府采购进口产品管理有关问题的通知》（财办库[2008]248号）：财政部门审核同意购买进口产品的，应当在采购文件中明确规定可以采购进口产品，但如果因信息不对称等原因，仍有满足需求的国内产品要求参与采购竞争的，采购人及其委托的采购代理机构不得对其加以限制，应当按照公平竞争原则实施采购。'
]

# 计算 embeddings
querys_emb = model.encode(querys, convert_to_tensor=True)
sentences_emb = model.encode(sentences, convert_to_tensor=True)

print('俩shape：')
print(querys_emb.shape)
print(sentences_emb.shape)
# 计算余弦相似度（直接使用 util.pytorch_cos_sim）
cos_sim = util.pytorch_cos_sim(querys_emb, sentences_emb)

print("余弦相似矩阵:")
print(cos_sim)
print("行最大和列最大索引：")
print(cos_sim.argmax(dim=-1))
print(cos_sim.argmax(dim=0))

