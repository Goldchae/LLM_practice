# 예제 3.30. 학습한 모델을 불러와 pipeline을 활용해 추론하기
# 실습을 새롭게 시작하는 경우 데이터셋 다시 불러오기 실행
import torch
import torch.nn.functional as F
from datasets import load_dataset

dataset = load_dataset("klue", "ynat", split="validation")



from transformers import pipeline

model_id = "Goldchea/roberta-base-klue-ynat-classification"

model_pipeline = pipeline("text-classification", model=model_id)

print(model_pipeline(dataset["title"][:5]))
'''
토크나이저와 모델을 결합해 데이터의 전후차이와 모델 추론을 간단하게 수행하는 pipeline
작업 종류, 모델, 설정 등을 인자로 넘겨받음
'''








# #예제 3.31. 커스텀 파이프라인 구현
# import torch
# from torch.nn.functional import softmax
# from transformers import AutoModelForSequenceClassification, AutoTokenizer

# class CustomPipeline:
#     def __init__(self, model_id):
#         self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#         self.model.eval()

#     def __call__(self, texts):
#         tokenized = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

#         with torch.no_grad():
#             outputs = self.model(**tokenized)
#             logits = outputs.logits

#         probabilities = softmax(logits, dim=-1)
#         scores, labels = torch.max(probabilities, dim=-1)
#         labels_str = [self.model.config.id2label[label_idx] for label_idx in labels.tolist()]

#         return [{"label": label, "score": score.item()} for label, score in zip(labels_str, scores)]

# custom_pipeline = CustomPipeline(model_id)
# custom_pipeline(dataset['title'][:5])
# '''
# - tokenizer를 통해 토큰을 수행
# - 모델 추론 수행 
# - 가장 큰 예측 확률을 갖는 클래스를 추출
# 결과 반환
# '''