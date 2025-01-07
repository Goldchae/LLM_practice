# 예제 3.17. 모델 학습에 사용할 연합뉴스 데이터셋 다운로드
from datasets import load_dataset
klue_tc_train = load_dataset('klue', 'ynat', split='train')
klue_tc_eval = load_dataset('klue', 'ynat', split='validation')
# klue_tc_train
# klue_tc_train[0]
# klue_tc_train.features['label'].names
# ['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치']





# 예제 3.18. 실습에 사용하지 않는 불필요한 컬럼 제거
klue_tc_train = klue_tc_train.remove_columns(['guid', 'url', 'date'])
klue_tc_eval = klue_tc_eval.remove_columns(['guid', 'url', 'date'])





# 예제 3.19. 카테고리를 문자로 표기한 label_str 컬럼 추가

#klue_tc_train.features['label']
# ClassLabel(names=['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치'], id=None)

#klue_tc_train.features['label'].int2str(1)
# '경제'

klue_tc_label = klue_tc_train.features['label']

def make_str_label(batch):
  batch['label_str'] = klue_tc_label.int2str(batch['label'])
  return batch

klue_tc_train = klue_tc_train.map(make_str_label, batched=True, batch_size=1000)

# klue_tc_train[0]
# {'title': '유튜브 내달 2일까지 크리에이터 지원 공간 운영', 'label': 3, 'label_str': '생활문화'}






#예제 3.20. 학습/검증/테스트 데이터셋 분할
train_dataset = klue_tc_train.train_test_split(test_size=10000, shuffle=True, seed=42)['test']
dataset = klue_tc_eval.train_test_split(test_size=1000, shuffle=True, seed=42)
test_dataset = dataset['test']
valid_dataset = dataset['train'].train_test_split(test_size=1000, shuffle=True, seed=42)['test']
