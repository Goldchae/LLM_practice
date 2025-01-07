
#예제 3.8. 토크나이저 불러오기
from transformers import AutoTokenizer
model_id = 'klue/roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)
'''
AutoTokenizer 클래스
- 모델-토크나이저 호환성 체크하기!
- 토크나이저도 학습 데이터로 "어휘 사전" 구축하기 때문에 모델과 함께 저장
'''


# 예제 3.9. 토크나이저 사용하기
tokenized = tokenizer("토크나이저는 텍스트를 토큰 단위로 나눈다")
print(tokenized)
# {'input_ids': [0, 9157, 7461, 2190, 2259, 8509, 2138, 1793, 2855, 5385, 2200, 20950, 2],
#  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

print(tokenizer.convert_ids_to_tokens(tokenized['input_ids']))
# ['[CLS]', '토크', '##나이', '##저', '##는', '텍스트', '##를', '토', '##큰', '단위', '##로', '나눈다', '[SEP]']

print(tokenizer.decode(tokenized['input_ids']))
# [CLS] 토크나이저는 텍스트를 토큰 단위로 나눈다 [SEP]

print(tokenizer.decode(tokenized['input_ids'], skip_special_tokens=True))
# 토크나이저는 텍스트를 토큰 단위로 나눈다
'''
tokenizer에 텍스트를 넣으면 반환하는 것
=
토큰 아이디 리스트(토크나이저 사전의 몇 번째 항목인지) input_ids
토큰이 실제 텍스트인지(1)/길이 맞추기용 패딩인지 알려주는 attention_mask
토큰이 속한 문장의 아이디(0 첫 번째 문장) token_type_ids

(skip_special_tokens 특수 토큰 제외)
CLS : 문장이나 입력 텍스트의 시작을 나타내는 토큰
SEP : 문장이나 입력 간의 구분을 나타내는 토큰

decode : 자연어 변환
'''



#예제 3.10. 토크나이저에 여러 문장 넣기
tokenizer(['첫 번째 문장', '두 번째 문장'])

# {'input_ids': [[0, 1656, 1141, 3135, 6265, 2], [0, 864, 1141, 3135, 6265, 2]],
# 'token_type_ids': [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
# 'attention_mask': [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]}


# 예제 3.11. 하나의 데이터에 여러 문장이 들어가는 경우 (괄호 두 개!)
tokenizer([['첫 번째 문장', '두 번째 문장']])

# {'input_ids': [[0, 1656, 1141, 3135, 6265, 2, 864, 1141, 3135, 6265, 2]],
# 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
# 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
'''
여러 개 데이터
여러 개 문장이 한 개의 데이터 -> 대괄호 2개로 감싸기
'''



# 예제 3.12. 토큰 아이디를 문자열로 복원
first_tokenized_result = tokenizer(['첫 번째 문장', '두 번째 문장'])['input_ids']
tokenizer.batch_decode(first_tokenized_result)
# ['[CLS] 첫 번째 문장 [SEP]', '[CLS] 두 번째 문장 [SEP]']

second_tokenized_result = tokenizer([['첫 번째 문장', '두 번째 문장']])['input_ids']
tokenizer.batch_decode(second_tokenized_result)
# ['[CLS] 첫 번째 문장 [SEP] 두 번째 문장 [SEP]']





#예제 3.13. BERT 토크나이저와 RoBERTa 토크나이저
bert_tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
bert_tokenizer([['첫 번째 문장', '두 번째 문장']])
# {'input_ids': [[2, 1656, 1141, 3135, 6265, 3, 864, 1141, 3135, 6265, 3]],
# 'token_type_ids': [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]],
# 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}

roberta_tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
roberta_tokenizer([['첫 번째 문장', '두 번째 문장']])
# {'input_ids': [[0, 1656, 1141, 3135, 6265, 2, 864, 1141, 3135, 6265, 2]],
# 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
# 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}

en_roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
en_roberta_tokenizer([['first sentence', 'second sentence']])
# {'input_ids': [[0, 9502, 3645, 2, 2, 10815, 3645, 2]],
# 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1]]}
'''
NSP : 문장이 서로 이어지는지 맞추는 작업
token_type_ids 필요
nsp가 없는 모델에는 token_type_ids 필요 없음
'''





#예제 3.14. attention_mask 확인
tokenizer(['첫 번째 문장은 짧다.', '두 번째 문장은 첫 번째 문장 보다 더 길다.'], padding='longest')

# {'input_ids': [[0, 1656, 1141, 3135, 6265, 2073, 1599, 2062, 18, 2, 1, 1, 1, 1, 1, 1],
# [0, 864, 1141, 3135, 6265, 2073, 1656, 1141, 3135, 6265, 3632, 831, 647, 2062, 18, 2]],
# 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
# [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
'''
padding 인자에 longest 를 넣어서
긴 문장을 기준으로 짧은 문장에 패딩 토큰을 채움 (attention_mask[0] 뒷부분)
'''