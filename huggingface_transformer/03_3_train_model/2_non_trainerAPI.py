# 예제 3.24. Trainer를 사용하지 않는 학습: (1) 학습을 위한 모델과 토크나이저 준비
# import torch
# from tqdm.auto import tqdm
# from torch.utils.data import DataLoader
# from transformers import AdamW

# def tokenize_function(examples): # 제목(title) 컬럼에 대한 토큰화
#     return tokenizer(examples["title"], padding="max_length", truncation=True)

# # 모델과 토크나이저 불러오기
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_id = "klue/roberta-base"
# model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=len(train_dataset.features['label'].names))
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model.to(device)
# 예제 3.25 Trainer를 사용하지 않는 학습: (2) 학습을 위한 데이터 준비
# def make_dataloader(dataset, batch_size, shuffle=True):
#     dataset = dataset.map(tokenize_function, batched=True).with_format("torch") # 데이터셋에 토큰화 수행
#     dataset = dataset.rename_column("label", "labels") # 컬럼 이름 변경
#     dataset = dataset.remove_columns(column_names=['title']) # 불필요한 컬럼 제거
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# # 데이터로더 만들기
# train_dataloader = make_dataloader(train_dataset, batch_size=8, shuffle=True)
# valid_dataloader = make_dataloader(valid_dataset, batch_size=8, shuffle=False)
# test_dataloader = make_dataloader(test_dataset, batch_size=8, shuffle=False)
# 예제 3.26. Trainer를 사용하지 않는 학습: (3) 학습을 위한 함수 정의
# def train_epoch(model, data_loader, optimizer):
#     model.train()
#     total_loss = 0
#     for batch in tqdm(data_loader):
#         optimizer.zero_grad()
#         input_ids = batch['input_ids'].to(device) # 모델에 입력할 토큰 아이디
#         attention_mask = batch['attention_mask'].to(device) # 모델에 입력할 어텐션 마스크
#         labels = batch['labels'].to(device) # 모델에 입력할 레이블
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels) # 모델 계산
#         loss = outputs.loss # 손실
#         loss.backward() # 역전파
#         optimizer.step() # 모델 업데이트
#         total_loss += loss.item()
#     avg_loss = total_loss / len(data_loader)
#     return avg_loss
# 예제 3.27. Trainer를 사용하지 않는 학습: (4) 평가를 위한 함수 정의
# def evaluate(model, data_loader):
#     model.eval()
#     total_loss = 0
#     predictions = []
#     true_labels = []
#     with torch.no_grad():
#         for batch in tqdm(data_loader):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)
#             outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#             logits = outputs.logits
#             loss = outputs.loss
#             total_loss += loss.item()
#             preds = torch.argmax(logits, dim=-1)
#             predictions.extend(preds.cpu().numpy())
#             true_labels.extend(labels.cpu().numpy())
#     avg_loss = total_loss / len(data_loader)
#     accuracy = np.mean(np.array(predictions) == np.array(true_labels))
#     return avg_loss, accuracy
# 예제 3.28 Trainer를 사용하지 않는 학습: (5) 학습 수행
# num_epochs = 1
# optimizer = AdamW(model.parameters(), lr=5e-5)

# # 학습 루프
# for epoch in range(num_epochs):
#     print(f"Epoch {epoch+1}/{num_epochs}")
#     train_loss = train_epoch(model, train_dataloader, optimizer)
#     print(f"Training loss: {train_loss}")
#     valid_loss, valid_accuracy = evaluate(model, valid_dataloader)
#     print(f"Validation loss: {valid_loss}")
#     print(f"Validation accuracy: {valid_accuracy}")

# # Testing
# _, test_accuracy = evaluate(model, test_dataloader)
# print(f"Test accuracy: {test_accuracy}") # 정확도 0.82