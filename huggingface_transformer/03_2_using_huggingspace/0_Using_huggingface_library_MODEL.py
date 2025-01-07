#예제 3.2. 모델 아이디로 모델 불러오기
from transformers import AutoModel
model_id = 'klue/roberta-base'
model = AutoModel.from_pretrained(model_id)
'''
Automodel 클래스
- 모델의 바디를 불러오는 클래스

Automodel.from_pretrained(모델 경로)
- 모델 경로 (웹 허브/ 로컬)에서 모델 불러옴

허깅페이스 모델의 config.json
- 모델의 종류, 설정 파라미터, 어휘 사전 크기, 사용하는 토크나이저 클래스 등의 정보가 저장됨
- Automodel 클래스가 참고하여 모델 등을 불러옴
'''

#예제 3.4. 분류 헤드가 포함된 모델 불러오기
from transformers import AutoModelForSequenceClassification
model_id = 'SamLowe/roberta-base-go_emotions'
classification_model = AutoModelForSequenceClassification.from_pretrained(model_id)
'''
AutoModelForSequenceClassification 클래스
- 텍스트 시퀀스 분류를 위한 헤드가 포함된 모델(+바디) 을 불러오는 클래스
'''

#예제 3.6. 분류 헤드가 랜덤으로 초기화된 모델 불러오기
from transformers import AutoModelForSequenceClassification
model_id = 'klue/roberta-base'
classification_model = AutoModelForSequenceClassification.from_pretrained(model_id)
'''
분류 헤드가 붙은 모델을 불러올 수 있는 AutoModelForSequenceClassification 클래스로 헤드가 없는 모델을(바디 파라미터만) 불러오면?
=> 경고 발생 ( 분류 헤더의 파라미터를 랜덤으로 초기화시킴/ 추가 학습해서 사용할 것)
'''