# 예제 10.1 문장 임베딩을 활용한 단어 간 유사도 계산
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

smodel = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
dense_embeddings = smodel.encode(['학교', '공부', '운동'])
cosine_similarity(dense_embeddings) # 코사인 유사도
# array([[1.0000001 , 0.5950744 , 0.32537547],
#       [0.5950744 , 1.0000002 , 0.54595673],
#       [0.32537547, 0.54595673, 0.99999976]], dtype=float32)



# 예제 10.2 원핫 인코딩의 한계
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

word_dict = {"school": np.array([[1, 0, 0]]),
"study": np.array([[0, 1, 0]]),
"workout": np.array([[0, 0, 1]])
}

# 두 단어 사이의 코사인 유사도 계산하기
cosine_school_study = cosine_similarity(word_dict["school"], word_dict['study']) # 0
cosine_school_workout = cosine_similarity(word_dict['school'], word_dict['workout']) # 0