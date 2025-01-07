예제 3.15. KLUE MRC 데이터셋 다운로드
from datasets import load_dataset
klue_mrc_dataset = load_dataset('klue', 'mrc')
# klue_mrc_dataset_only_train = load_dataset('klue', 'mrc', split='train')
예제 3.16. 로컬의 데이터 활용하기
(안내) 아래 코드를 실행하기 위해서는 구글 코랩에 csv 파일이 업로드 되어야 합니다. 허깅페이스 datasets 형식으로 쉽게 변환할 수 있다는 점을 보여주기 위한 예시 코드입니다.

from datasets import load_dataset
# 로컬의 데이터 파일을 활용
dataset = load_dataset("csv", data_files="my_file.csv")

# 파이썬 딕셔너리 활용
from datasets import Dataset
my_dict = {"a": [1, 2, 3]}
dataset = Dataset.from_dict(my_dict)

# 판다스 데이터프레임 활용
from datasets import Dataset
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3]})
dataset = Dataset.from_pandas(df)