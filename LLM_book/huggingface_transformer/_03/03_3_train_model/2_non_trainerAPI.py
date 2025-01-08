# 예제 3.24. Trainer를 사용하지 않는 학습: (1) 학습을 위한 모델과 토크나이저 준비
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW

def tokenize_function(examples): # 제목(title) 컬럼에 대한 토큰화
    return tokenizer(examples["title"], padding="max_length", truncation=True)

# 모델과 토크나이저 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "klue/roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=len(train_dataset.features['label'].names))
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.to(device)
'''
model.to(device)
- 기존 train()이 해주던 GPU로의 모델 이동을 직접 수행
'''






# 예제 3.25 Trainer를 사용하지 않는 학습: (2) 학습을 위한 데이터 준비
def make_dataloader(dataset, batch_size, shuffle=True):
    dataset = dataset.map(tokenize_function, batched=True).with_format("torch") # 데이터셋에 토큰화 수행
    dataset = dataset.rename_column("label", "labels") # 컬럼 이름 변경
    dataset = dataset.remove_columns(column_names=['title']) # 불필요한 컬럼 제거
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# 데이터로더 만들기
train_dataloader = make_dataloader(train_dataset, batch_size=8, shuffle=True)
valid_dataloader = make_dataloader(valid_dataset, batch_size=8, shuffle=False)
test_dataloader = make_dataloader(test_dataset, batch_size=8, shuffle=False)
'''
전처리
- 제목 컬럼에 대한 토큰화 진행
- rename_column 메서드를 통한 컬럼명 재정의
- remove_column으로 컬럼 삭제
- pytorch의 Dataloader 클래스를 이용해 데이터셋을 배치 데이터로 변환


( trainer api는 토큰화를 제외한 나머지 과정을 알아서 처리해줌)
'''






# 예제 3.26. Trainer를 사용하지 않는 학습: (3) 학습을 위한 함수 정의
def train_epoch(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device) # 모델에 입력할 토큰 아이디
        attention_mask = batch['attention_mask'].to(device) # 모델에 입력할 어텐션 마스크
        labels = batch['labels'].to(device) # 모델에 입력할 레이블
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels) # 모델 계산
        loss = outputs.loss # 손실
        loss.backward() # 역전파
        optimizer.step() # 모델 업데이트
        total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss
'''
학습을 위한 함수
train() 메서드
- 모델을 학습 모드로 변경


데이터 로더 내부 배치 데이터
- input_ids : 토큰 아이디
- attention_mask : 어텐션 마스크
- labels : 정답 레이블
=> model에 인자로 전달하여 모델 계산 수행



'모델 계산 결과'에는 '레이블과의 차이를 통해 계산된 손실'이 있음
손실값을 이용해 '역전파' 수행
역전파 결과값을 바탕으로 모델 업데이트 수행
'''









# 예제 3.27. Trainer를 사용하지 않는 학습: (4) 평가를 위한 함수 정의
def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    return avg_loss, accuracy
'''
모델 평가를 위한 함수
eval() 메서드
- 모델을 추론 모드로 변경


모델 계산 결과의 logits 속성에서 가장 큰 값으로 예측한 카테고리 정보 찾기


손실도/정확도 확인 가능'''









# 예제 3.28 Trainer를 사용하지 않는 학습: (5) 학습 수행
num_epochs = 1
optimizer = AdamW(model.parameters(), lr=5e-5)

# 학습 루프
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss = train_epoch(model, train_dataloader, optimizer)
    print(f"Training loss: {train_loss}")
    valid_loss, valid_accuracy = evaluate(model, valid_dataloader)
    print(f"Validation loss: {valid_loss}")
    print(f"Validation accuracy: {valid_accuracy}")

# Testing
_, test_accuracy = evaluate(model, test_dataloader)
print(f"Test accuracy: {test_accuracy}") # 정확도 0.82
'''
정의한 학습 함수/ 성능 평가 함수로 학습 진행

- 에포크 수 1 설정
- AdamW 옵티마이저 사용
'''









'''
옵티마이저 (Optimizer)

신경망 학습에서 손실 함수(Loss Function)를 최소화하기 위해 모델의 가중치(Weights)와 편향(Biases)을 업데이트하는 알고리즘

주요 옵티마이저 알고리즘:

SGD (Stochastic Gradient Descent): 기초적인 경사 하강법으로, 데이터의 일부(배치)를 이용해 가중치를 갱신
Adam (Adaptive Moment Estimation): SGD의 확장 버전으로, 학습률을 자동으로 조정하며, 빠르고 안정적인 수렴을 제공
RMSProp, Adagrad: 학습률을 데이터의 스케일에 따라 조정하는 방법들
에포크 (Epoch)

에포크는 신경망이 훈련 데이터를 한 번 모두 사용하여 학습을 완료하는 주기

데이터셋이 크면 한 번에 모든 데이터를 사용하기 어려워 미니배치(Mini-batch)로 나누어 학습하며, 배치를 여러 번 반복하는 것이 에포크 하나를 구성
예: 데이터셋이 10,000개이고 배치 크기가 100이라면, 한 에포크는 100번의 배치 업데이트로 구성

역전파 (Backpropagation)

오차를 신경망의 입력 방향으로 전파하며 가중치를 업데이트하는 알고리즘

순전파(Forward Propagation): 입력 데이터를 신경망의 각 계층을 통과시켜 출력값을 계산
역전파: 출력값과 목표값(라벨) 사이의 오차를 계산하여, 이 오차가 네트워크를 거꾸로 전파되면서 가중치를 조정
배치(Batch)

딥러닝과 머신러닝에서 모델 학습 시 데이터를 처리하는 단위

데이터셋이 너무 크거나 연산량이 많아 한 번에 전체 데이터를 처리하기 어려운 경우, 데이터를 작은 그룹(배치)으로 나누어 처리'''