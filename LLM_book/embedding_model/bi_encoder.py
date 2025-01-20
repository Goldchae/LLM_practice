# 예제 10.3 Sentence-Transformers 라이브러리로 바이 인코더 생성하기
'''
< 진행 로직 >

1. 텍스트 입력

2. Bert 모델 연산 처리

3. 문맥이 반영된 단어 임베딩들 출력 (Bert 모델은 입력하는 토큰마다 각각 출력 임베딩을 생성)

4. 풀링 층을 통과하여 여러 단어 임베딩들을 하나의 임베딩으로 통합 ( 문장의 길이가 다르면 유사도 계산이 쉽지 않으므로 )

이후 코사인 유사도 등을 이용하여 두 문장 사이의 임베딩 거리 계산
'''
from sentence_transformers import SentenceTransformer, models

# 사용할 BERT 모델
word_embedding_model = models.Transformer('klue/roberta-base')

# 풀링 층 차원 입력하기
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

# 두 모듈 결합하기
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

'''
klue/roberta-base에서 모델 가져오기
Pooling 클래스로 풀링 층 생성하기  
get_word_embedding_dimension() 메서드로 입력으로 들어오는 토큰 임베딩의 차원을 알려줌
SentenceTransformer 클래스에 모듈로 언어 모델과 풀링 층을 입력해서 바이 인코더를 생성


< 3가지 풀링 모드 >

클래스 모드 (pooling_mode_cls_tokens)
Bert 모델의 첫 번째 토큰인 [CLS] 토큰의 출력 임베딩을 문장 임베딩으로 사용
평균 모드 (pooling_mode_mean_tokens)
Bert 모델에서 모든 입력 토큰의 출력 임베딩을 평균한 값을 문장 임베딩으로 사용
최대 모드 (pooling_mode_max_tokens)
Bert 모델에서 모든 입력 토큰의 출력 임베딩에서 문장 길이 방향 최댓값을 문장 임베딩으로 사용
'''

# 예제 10.4 코드로 살펴보는 평균 모드
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

#예제 10.5 코드로 살펴보는 최대 모드
def max_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9
    return torch.max(token_embeddings, 1)[0]