#예제 10.18 상호 순위 조합 함수 구현
from collections import defaultdict

def reciprocal_rank_fusion(rankings:List[List[int]], k=5):
    rrf = defaultdict(float)
    for ranking in rankings:
        for i, doc_id in enumerate(ranking, 1):
            rrf[doc_id] += 1.0 / (k + i)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)












#예제 10.19 예시 데이터에 대한 상호 순위 조합 결과 확인하기
rankings = [[1, 4, 3, 5, 6], [2, 1, 3, 6, 4]]
reciprocal_rank_fusion(rankings)

# [(1, 0.30952380952380953),
#  (3, 0.25),
#  (4, 0.24285714285714285),
#  (6, 0.2111111111111111),
#  (2, 0.16666666666666666),
#  (5, 0.1111111111111111)]