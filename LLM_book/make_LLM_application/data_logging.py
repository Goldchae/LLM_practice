# 9.4절 데이터 로깅
# 예제 9.15 W&B에 로그인하기
import os
import wandb

wandb.login()
wandb.init(project="trace-example")

"""

💴 데이터 로깅
사용자의 입력과 LLM 생성 출력을 기록하는 데이터 로깅

 

동일한 입력에 대해서도 LLM의 응답이 매번 달라지기 때문에 

기록 필요 

 

 

🗒️ 대표적인 로깅 도구🗒️

- W&B

 
Weights & Biases: The AI Developer Platform

Weights & Biases는 모델을 학습 및 파인튜닝하고, 실험부터 생산까지 모델을 관리하며, LLM으로 구동되는 GenAI 애플리케이션을 추적 및 평가할 수 있는 선도적인 AI 개발자 플랫폼입니다.

site.wandb.ai
- MLflow

 
MLflow | MLflow

Description will go into a meta tag in <head />

mlflow.org
- PromptLayer

 
PromptLayer - The cleanest way to prompt engineer. Platform for prompt management, prompt evaluations, and LLM observability

Best practices Prompt Management and collaboration using a CMS Mar 7, 2024

www.promptlayer.com
 

 

 

🗒️ 로깅 도구 W&B 써보기

LLM 입력, 출력, 시간, 에러 유무, 등 알 수 있음 

https://wandb.ai/authorize

api 키 찾기
"""





# 예제 9.16 OpenAI API 로깅하기
import datetime
from openai import OpenAI
from wandb.sdk.data_types.trace_tree import Trace

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

system_message = "You are a helpful assistant."
query = "대한민국의 수도는 어디야?"
temperature = 0.2
model_name = "gpt-3.5-turbo"

response = client.chat.completions.create(model=model_name,
                                        messages=[{"role": "system", "content": system_message},{"role": "user", "content": query}],
                                        temperature=temperature
                                        )

root_span = Trace(
      name="root_span",
      kind="llm",
      status_code="success",
      status_message=None,
      metadata={"temperature": temperature,
                "token_usage": dict(response.usage),
                "model_name": model_name},
      inputs={"system_prompt": system_message, "query": query},
      outputs={"response": response.choices[0].message.content},
      )

root_span.log(name="openai_trace")
'''
Trace 클래스 : 

openAI 요청의 입력과 생성 결과, 생성 성공 여부, 생성에 사용한 설정값 등을 이용해 기록할 데이터 생성

 

Trace 클래스의 log 메서드 :

로그를 W&B에 저장 (각 요청 구분을 위한 이름 설정도 가능)
'''









# 예제 9.17 라마인덱스 W&B 로깅
from datasets import load_dataset
import llama_index
from llama_index.core import Document, VectorStoreIndex, ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.core import set_global_handler
# 로깅을 위한 설정 추가
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
set_global_handler("wandb", run_args={"project": "llamaindex"})
wandb_callback = llama_index.core.global_handler
service_context = ServiceContext.from_defaults(llm=llm)

dataset = load_dataset('klue', 'mrc', split='train')
text_list = dataset[:100]['context']
documents = [Document(text=t) for t in text_list]

index = VectorStoreIndex.from_documents(documents, service_context=service_context)

print(dataset[0]['question']) # 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?

query_engine = index.as_query_engine(similarity_top_k=1, verbose=True)
response = query_engine.query(
    dataset[0]['question']
)
'''
set_global_handler 함수 :

라마인덱스 내부에서 W&B에 로그를 전송하도록 설정
'''
