# 예제 10.20 하이브리드 검색 구현하기
def dense_vector_search(query:str, k:int):
  query_embedding = sentence_model.encode([query])
  distances, indices = index.search(query_embedding, k)
  return distances[0], indices[0]

def hybrid_search(query, k=20):
  _, dense_search_ranking = dense_vector_search(query, 100)
  _, bm25_search_ranking = bm25.get_top_k(query, 100)

  results = reciprocal_rank_fusion([dense_search_ranking, bm25_search_ranking], k=k)
  return results











# 예제 10.21 예시 데이터에 대한 하이브리드 검색 결과 확인
query = "이번 연도에는 언제 비가 많이 올까?"
print("검색 쿼리 문장: ", query)
results = hybrid_search(query)
for idx, score in results[:3]:
  print(klue_mrc_dataset['context'][idx][:50])

print("=" * 80)
query = klue_mrc_dataset[3]['question'] # 로버트 헨리 딕이 1946년에 매사추세츠 연구소에서 개발한 것은 무엇인가?
print("검색 쿼리 문장: ", query)

results = hybrid_search(query)
for idx, score in results[:3]:
  print(klue_mrc_dataset['context'][idx][:50])

# 출력 결과
# 검색 쿼리 문장:  이번 연도에는 언제 비가 많이 올까?
# 올여름 장마가 17일 제주도에서 시작됐다. 서울 등 중부지방은 예년보다 사나흘 정도 늦은  (정답)
# 갤럭시S5 언제 발매한다는 건지언제는 “27일 판매한다”고 했다가 “이르면 26일 판매한다  (오답)
# 연구 결과에 따르면, 오리너구리의 눈은 대부분의 포유류보다는 어류인 칠성장어나 먹장어, 그 (오답)
# ================================================================================
# 검색 쿼리 문장:  로버트 헨리 딕이 1946년에 매사추세츠 연구소에서 개발한 것은 무엇인가?
# 미국 세인트루이스에서 태어났고, 프린스턴 대학교에서 학사 학위를 마치고 1939년에 로체스 (정답)
# 1950년대 말 매사추세츠 공과대학교의 동아리 테크모델철도클럽에서 ‘해커’라는 용어가 처음 (오답)
# 1950년대 말 매사추세츠 공과대학교의 동아리 테크모델철도클럽에서 ‘해커’라는 용어가 처음 (오답)