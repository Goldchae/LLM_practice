from transformers import AutoTokenizer, AutoModel

text = "What is Huggingface Transformers?"
# BERT 모델 활용
bert_model = AutoModel.from_pretrained("bert-base-uncased")
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encoded_input = bert_tokenizer(text, return_tensors='pt')
bert_output = bert_model(**encoded_input)
print(bert_output) # => 각 토큰에 대한 벡터 표현 (텍스트 이해를 위한 고차원 임베딩)
# GPT-2 모델 활용
gpt_model = AutoModel.from_pretrained('gpt2')
gpt_tokenizer = AutoTokenizer.from_pretrained('gpt2')
encoded_input = gpt_tokenizer(text, return_tensors='pt')
gpt_output = gpt_model(**encoded_input) 
print(gpt_output) # => 각 토큰에 대한 다음 토큰 예측을 위한 임베딩 벡터 (텍스트 생성의 기초)