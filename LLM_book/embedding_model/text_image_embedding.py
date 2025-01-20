# 예제 10.6 한국어 문장 임베딩 모델로 입력 문장 사이의 유사도 계산
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

embs = model.encode(['잠이 안 옵니다',
                     '졸음이 옵니다',
                     '기차가 옵니다'])

cos_scores = util.cos_sim(embs, embs)
print(cos_scores)
# tensor([[1.0000, 0.6410, 0.1887],
#         [0.6410, 1.0000, 0.2730],
#         [0.1887, 0.2730, 1.0000]])

"""
한국어 문장 임베딩 모델 snunlp/KR-SBERT-V40K-klueNLI-augSTS
util.cos_sim으로 문장 사이의 코사인 유사도 계산
 

문장 사이의 유사도를 확인 가능 (1-0)
"""


#예제 10.7 CLIP 모델을 활용한 이미지와 텍스트 임베딩 유사도 계산
from PIL import Image
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('clip-ViT-B-32')

img_embs = model.encode([Image.open('dog.jpg'), Image.open('cat.jpg')])
text_embs = model.encode(['A dog on grass', 'Brown cat on yellow background'])

cos_scores = util.cos_sim(img_embs, text_embs)
print(cos_scores)
# tensor([[0.2771, 0.1509],
#         [0.2071, 0.3180]])

"""
clip-ViT-B-32 모델 : OpenAI가 개발한 텍스트-이미지 멀티 모달 모델
이미지와 텍스트의 임베딩을 동일한 벡터 공간의 임베딩으로 변환함 (서로 유사도 계산 가능)
"""