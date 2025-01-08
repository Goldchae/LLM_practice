#예제 3.29. 허깅페이스 허브에 모델 업로드
# 모델의 예측 아이디와 문자열 레이블을 연결할 데이터를 모델 config에 저장
id2label = {i: label for i, label in enumerate(train_dataset.features['label'].names)}
label2id = {label: i for i, label in id2label.items()}
model.config.id2label = id2label
model.config.label2id = label2id

from huggingface_hub import login

login(token="본인의 허깅페이스 토큰 입력")
repo_id = f"Goldchae/roberta-base-klue-ynat-classification"
# Trainer를 사용한 경우
trainer.push_to_hub(repo_id)
# 직접 학습한 경우
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)





# 모델 업로드
# https://huggingface.co/Goldchea/roberta-base-klue-ynat-classification