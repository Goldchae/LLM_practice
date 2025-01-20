# 예제 10.14 BM25 클래스 구현
import math
import numpy as np
from typing import List
from transformers import PreTrainedTokenizer
from collections import defaultdict

class BM25:
  def __init__(self, corpus:List[List[str]], tokenizer:PreTrainedTokenizer):
    self.tokenizer = tokenizer
    self.corpus = corpus
    self.tokenized_corpus = self.tokenizer(corpus, add_special_tokens=False)['input_ids']
    self.n_docs = len(self.tokenized_corpus)
    self.avg_doc_lens = sum(len(lst) for lst in self.tokenized_corpus) / len(self.tokenized_corpus)
    self.idf = self._calculate_idf()
    self.term_freqs = self._calculate_term_freqs()

  def _calculate_idf(self):
    idf = defaultdict(float)
    for doc in self.tokenized_corpus:
      for token_id in set(doc):
        idf[token_id] += 1
    for token_id, doc_frequency in idf.items():
      idf[token_id] = math.log(((self.n_docs - doc_frequency + 0.5) / (doc_frequency + 0.5)) + 1)
    return idf

  def _calculate_term_freqs(self):
    term_freqs = [defaultdict(int) for _ in range(self.n_docs)]
    for i, doc in enumerate(self.tokenized_corpus):
      for token_id in doc:
        term_freqs[i][token_id] += 1
    return term_freqs

  def get_scores(self, query:str, k1:float = 1.2, b:float=0.75):
    query = self.tokenizer([query], add_special_tokens=False)['input_ids'][0]
    scores = np.zeros(self.n_docs)
    for q in query:
      idf = self.idf[q]
      for i, term_freq in enumerate(self.term_freqs):
        q_frequency = term_freq[q]
        doc_len = len(self.tokenized_corpus[i])
        score_q = idf * (q_frequency * (k1 + 1)) / ((q_frequency) + k1 * (1 - b + b * (doc_len / self.avg_doc_lens)))
        scores[i] += score_q
    return scores

  def get_top_k(self, query:str, k:int):
    scores = self.get_scores(query)
    top_k_indices = np.argsort(scores)[-k:][::-1]
    top_k_scores = scores[top_k_indices]
    return top_k_scores, top_k_indices
  








# 예제 10.15 BM25 점수 계산 확인해 보기
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')

bm25 = BM25(['안녕하세요', '반갑습니다', '안녕 서울'], tokenizer)
bm25.get_scores('안녕')
# array([0.44713859, 0.        , 0.52354835])










#예제 10.16 BM25 검색 결과의 한계
# BM25 검색 준비
bm25 = BM25(klue_mrc_dataset['context'], tokenizer)

query = "이번 연도에는 언제 비가 많이 올까?"
_, bm25_search_ranking = bm25.get_top_k(query, 100)

for idx in bm25_search_ranking[:3]:
  print(klue_mrc_dataset['context'][idx][:50])

# 출력 결과
# 갤럭시S5 언제 발매한다는 건지언제는 “27일 판매한다”고 했다가 “이르면 26일 판매한다 (오답)
# 인구 비율당 노벨상을 세계에서 가장 많이 받은 나라, 과학 논문을 가장 많이 쓰고 의료 특 (오답)
# 올여름 장마가 17일 제주도에서 시작됐다. 서울 등 중부지방은 예년보다 사나흘 정도 늦은  (정답)












# 예제 10.17 BM25 검색 결과의 장점
query = klue_mrc_dataset[3]['question']  # 로버트 헨리 딕이 1946년에 매사추세츠 연구소에서 개발한 것은 무엇인가?
_, bm25_search_ranking = bm25.get_top_k(query, 100)

for idx in bm25_search_ranking[:3]:
  print(klue_mrc_dataset['context'][idx][:50])

# 출력 결과
# 미국 세인트루이스에서 태어났고, 프린스턴 대학교에서 학사 학위를 마치고 1939년에 로체스 (정답)
# ;메카동(メカドン)                                                      (오답)
# :성우 : 나라하시 미키(ならはしみき)
# 길가에 버려져 있던 낡은 느티나
# ;메카동(メカドン)                                                      (오답)
# :성우 : 나라하시 미키(ならはしみき)