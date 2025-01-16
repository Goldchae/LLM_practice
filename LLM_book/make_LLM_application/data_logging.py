# 9.4절 데이터 로깅
# 예제 9.15 W&B에 로그인하기
import os
import wandb

wandb.login()
wandb.init(project="trace-example")









# 예제 9.16 OpenAI API 로깅하기
import datetime
from openai import OpenAI
from wandb.sdk.data_types.trace_tree import Trace

client = OpenAI()
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