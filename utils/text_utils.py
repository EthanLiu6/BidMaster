from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import logging
from scripts import config

__all__ = ['TextUtils']


class TextUtils:

    def __init__(
            self,
            save_db_path,
            collection_name,
            encode_model: str,
            metric_type='COSINE',
            consistency_level='Strong'
    ):
        self.collection_name = collection_name
        self._save_db_path = save_db_path  # TODO： 可设计配置文件
        self._milvus_client = MilvusClient(uri=self._save_db_path, metric_type=metric_type,
                                           consistency_level=consistency_level)  # TODO： 可设计配置文件
        self._encode_model = SentenceTransformer(encode_model, device=config.device)

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
        logging.info(f"原文档长度：{origin_texts_length}")
        logging.info(f"预切分文本长度：{per_text_length}")
        logging.info(f"切分覆盖长度：{cover_length}")

        assert per_text_length < origin_texts_length, f"预切分长度{per_text_length}大于原始文本字符串长度{origin_texts_length}"
        assert per_text_length > cover_length, f'覆盖长度{per_text_length}大于了预切分长度{cover_length}'

        chunk_texts_list = []
        for idx in range(0, origin_texts_length, per_text_length):
            if idx == 0:  # 第一个切块不做覆盖
                chunk_texts_list.append(origin_texts[: per_text_length])
            else:
                chunk_texts_list.append(origin_texts[idx - cover_length: idx + per_text_length])

        return chunk_texts_list

    def text2vector(self, texts: list[str] | str):
        texts_embeddings = self._encode_model.encode(texts)
        embeddings_dim = len(texts_embeddings[0])
        return texts_embeddings, embeddings_dim

    def _create_tables(self, collection_name, embeddings_dim):
        self._milvus_client.create_collection(collection_name=collection_name, dimension=embeddings_dim)

    def insert_vector(self,
                      chunk_texts_list,
                      texts_embeddings,
                      embedding_dim: int,
                      from_doc=None,
                      ):
        # NOTE：目前使用的是Milvus向量数据库
        """
        :param chunk_texts_list:
        :param texts_embeddings:
        :param embedding_dim: 文本向量维度，必须与texts_embeddings维度相同
        :param from_doc: 文档来源（或者路径）
        :return:
        """

        ture_emb_dim = len(texts_embeddings[0])
        assert embedding_dim == ture_emb_dim, f'向量维度不匹配：{embedding_dim}!={ture_emb_dim}'
        logging.info(f'当前向量存储维度：{embedding_dim}')
        data = []
        idx = 0
        for sentence, embedding in zip(chunk_texts_list, texts_embeddings):
            data.append({"id": idx, "sentence": sentence, "vector": embedding, "from_doc": from_doc})
            idx += 1
        logging.info(f'向量库集合名：{self.collection_name}')
        if self._milvus_client.has_collection(self.collection_name):
            self._milvus_client.drop_collection(self.collection_name)

        self._create_tables(self.collection_name, embedding_dim)


        self._milvus_client.insert(collection_name=self.collection_name, data=data)

    def search_from_milvus_client(
            self,
            question: str,
            collection_name: str,
            limit=3
    ):
        search_res = self._milvus_client.search(
            collection_name=collection_name,
            data=[
                self._encode_model.encode(question)
            ],

            limit=limit,  # Return top limit results
            search_params={"metric_type": "IP", "params": {}},  # Inner product distance
            output_fields=["sentence", "from_docs"],  # Return the sentence field
        )

        return search_res
