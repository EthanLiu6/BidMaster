from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient


class TextTool:
    def __init__(
            self,
            save_db_path,
            encode_model: str,
            metric_type='IP',
            consistency_level='Strong'
    ):
        self.save_db_path = save_db_path  # TODO： 可设计配置文件
        self.milvus_client = MilvusClient(uri=self.save_db_path, metric_type=metric_type,
                                          consistency_level=consistency_level)  # TODO： 可设计配置文件
        self.encode_model = SentenceTransformer(encode_model)  # TODO： 可设计配置文件

    def chunk_texts(
            self,
            origin_texts: str,
            per_text_length: int,
            cover_length: int
    ) -> list[str]:
        """
        :param origin_texts: 读取并处理后的文本字符串
        :param per_text_length: 预切分的长度
        :param cover_length: 切分后一个与前一个的覆盖长度
        :return: 切分后切分文本列表
        """
        origin_texts_length = len(origin_texts)
        assert per_text_length > origin_texts_length, f"预切分长度{per_text_length}大于原始文本字符串长度{origin_texts_length}"
        assert per_text_length < cover_length, f'覆盖长度{per_text_length}大于了预切分长度{cover_length}'

        chunk_texts_list = []
        for idx in range(0, origin_texts_length, per_text_length):
            if idx == 0:  # 第一个切块不做覆盖
                chunk_texts_list.append(origin_texts[: per_text_length])
            else:
                chunk_texts_list.append(origin_texts[idx - cover_length, : idx + per_text_length])

        return chunk_texts_list

    def text2vector(self, texts: list[str] | str):
        texts_embeddings = self.encode_model.encode(texts)
        embeddings_dim = len(texts_embeddings[0])
        return texts_embeddings, embeddings_dim

    def insert_vector(self,
                      chunk_texts_list,
                      texts_embeddings,
                      collection_name,
                      embedding_dim: int,
                      fields: list[str] = ['id', 'sentence', 'vector', 'from_doc'],
                      ):
        # NOTE：目前使用的是Milvus向量数据库
        """
        :param save_db_path: 存放数据库路径以及名称 eg. '../../knowledge/vector_knowledge/laws.db'
        :param chunk_texts_list:
        :param texts_embeddings:
        :param collection_name: 类似结构化数据库的表名（暂定）
        :param embedding_dim: 文本向量维度，必须与texts_embeddings维度相同
        :param fields: list类型的字符串序列，是字段名称
        :param metric_type: milvus框架所计算相似度方式（具体查官网）
        :param consistency_level: Supported values are (`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`). See https://milvus.io/docs/consistency.md#Consistency-Level for more details.
        :return:
        """

        ture_emb_dim = len(texts_embeddings[0])
        assert embedding_dim == ture_emb_dim, f'向量维度不匹配:{embedding_dim}!={ture_emb_dim}'
        assert fields == ['id', 'sentence', 'vector', 'from_doc'], "目前还未提供别的字段插入设计"

        data = []
        idx = 0
        for sentence, embedding in zip(chunk_texts_list, texts_embeddings):
            data.append({"id": idx, "sentence": sentence, "vector": embedding, "from_doc": None})
            idx += 1

        self.milvus_client.insert(collection_name=collection_name, data=data)

    def search_from_milvus_client(
            self,
            question: str,
            collection_name: str,
            limit=3
    ):
        search_res = self.milvus_client.search(
            collection_name=collection_name,
            data=[
                self.encode_model.encode(question)
            ],

            limit=limit,  # Return top limit results
            search_params={"metric_type": "IP", "params": {}},  # Inner product distance
            output_fields=["sentence"],  # Return the sentence field
        )

        return search_res
