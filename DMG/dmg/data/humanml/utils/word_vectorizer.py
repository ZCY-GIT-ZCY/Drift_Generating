"""
DMG HumanML3D Utils - WordVectorizer
复用 MLD 的词向量器
"""

import pickle
import numpy as np


class WordVectorizer:
    """
    词向量器

    用于将文本 token 转换为词嵌入和位置 one-hot 向量
    """

    def __init__(self, vec_file, w2v_file):
        """
        初始化词向量器

        Args:
            vec_file: 词向量文件路径 (.p)
            w2v_file: 词到向量的映射文件路径 (.p)
        """
        with open(vec_file, 'rb') as f:
            self.data = pickle.load(f)
        with open(w2v_file, 'rb') as f:
            self.word2index = pickle.load(f)

        self.vector_size = 300  # GloVe 词向量维度

    def __len__(self):
        return len(self.data)

    def __call__(self, word):
        """
        将单词转换为词嵌入和位置 one-hot

        Args:
            word: 单词字符串

        Returns:
            word_emb: 词嵌入向量 [300]
            pos_oh: 位置 one-hot 向量 [15]
        """
        # 获取词嵌入
        word_emb = self.get_word_embedding(word)

        # 获取位置 one-hot
        pos_oh = self.get_pos_one_hot(word)

        return word_emb, pos_oh

    def get_word_embedding(self, word):
        """
        获取词嵌入

        Args:
            word: 单词字符串

        Returns:
            word_emb: 词嵌入向量
        """
        # 处理带词性的词（如 "walk/v" -> "walk"）
        if '/' in word:
            word = word.split('/')[0]

        # 小写
        word = word.lower()

        # 获取向量
        if word in self.word2index:
            index = self.word2index[word]
            word_emb = self.data[index]
        else:
            # 如果词不在词汇表中，返回零向量
            word_emb = np.zeros(self.vector_size, dtype=np.float32)

        return word_emb.astype(np.float32)

    def get_pos_one_hot(self, word):
        """
        获取位置 one-hot 向量

        Args:
            word: 单词字符串

        Returns:
            pos_oh: 位置 one-hot 向量 [15]
        """
        # 提取词性
        if '/' in word:
            pos = word.split('/')[1] if len(word.split('/')) > 1 else 'OTHER'
        else:
            pos = 'OTHER'

        # POS 标签到索引的映射
        pos_map = {
            'VERB': 0,
            'NOUN': 1,
            'DET': 2,
            'ADP': 3,
            'NUM': 4,
            'CCONJ': 5,
            'ADV': 6,
            'PRON': 7,
            'AUX': 8,
            'PART': 9,
            'INTJ': 10,
            'PUNCT': 11,
            'SCONJ': 12,
            'X': 13,
            'OTHER': 14,
        }

        pos_oh = np.zeros(15, dtype=np.float32)
        pos_idx = pos_map.get(pos, 14)  # 默认 OTHER
        pos_oh[pos_idx] = 1.0

        return pos_oh
