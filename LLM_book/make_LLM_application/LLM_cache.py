# 9.2절 LLM 캐시
# 예제 9.6 실습에 사용할 OpenAI와 크로마 클라이언트 생성
import os
import chromadb
from openai import OpenAI

api_key = os.environ["OPENAI_API_KEY"] = "자신의 OpenAI API 키 입력"

openai_client = OpenAI(api_key = api_key)
chroma_client = chromadb.Client()


"""
💴 LLM 캐시
LLM 생성(추론) => 많은 시간과 비용

가능하면 최대한 줄여야 함

 

요청과 생성 결과를 기록하고 이후에 동일하거나 비슷한 요청이 들어오면 새롭게 텍스트를 생성하지 않고 이전의 생성 결과를 가져와 응답함으로써 LLM 응답 시간을 줄임 

 

 

벡터 데이터베이스를 이용해서 LLM 캐시 직접 구현해 보기

💶 LLM 캐시 작동 원리
캐시 요청을 통해 기존에 동일하거나 유사한 요청이 있었는지 확인하고 만약 있었다면 LLM 캐시에 저장된 답변을 모델에 전달

 

💵 일치 캐시 (Exact match)

문자열이 완전히 동일한 응답에 대한 캐시 사용

딕셔너리 형식 자료구조에서 키가 동일한 프롬프트를 찾아 값을 반환

 

💵 유사 검색 캐시 (Similar search)

문자열이 유사한 응답에 대한 캐시 사용

 

요청을 임베딩 모델을 이용해 임베딩 벡터로 변환,

벡터 데이터베이스에 유사한 요청이 있는지 검색,

유사한 벡터가 있다면 반환,

없다면 LLM 요청 생성하고 벡터 DB에 새롭게 저장

🥞 실습 : OpenAi api cache 구현
Chroma : 오픈 소스 벡터 데이터베이스

 

time.time()
소요 시간 체크하기
"""





# 예제 9.7 LLM 캐시를 사용하지 않았을 때 동일한 요청 처리에 걸린 시간 확인
import time

def response_text(openai_resp):
    return openai_resp.choices[0].message.content

question = "북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?"
for _ in range(2):
    start_time = time.time()
    response = openai_client.chat.completions.create(
      model='gpt-3.5-turbo',
      messages=[
        {
            'role': 'user',
            'content': question
        }
      ],
    )
    response = response_text(response)
    print(f'질문: {question}')
    print("소요 시간: {:.2f}s".format(time.time() - start_time))
    print(f'답변: {response}\n')

# 질문: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?
# 소요 시간: 2.71s
# 답변: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은 겨울 시즌인 11월부터 다음 해 3월까지입니다. 이 기간 동안 기온이 급격히 하락하며 한반도에 한기가 밀려오게 됩니다.

# 질문: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?
# 소요 시간: 4.13s
# 답변: 북태평양 기단과 오호츠크해 기단은 겨울에 만나 국내에 머무르는 것이 일반적입니다. 이 기단들은 주로 11월부터 2월이나 3월까지 국내에 영향을 미치며, 한국의 겨울철 추위와 함께 한반도 전역에 형성되는 강한 서북풍과 냉기를 가져옵니다.









# 예제 9.8 파이썬 딕셔너리를 활용한 일치 캐시 구현
class OpenAICache:
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.cache = {}

    def generate(self, prompt):
        if prompt not in self.cache:
            response = self.openai_client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
            )
            self.cache[prompt] = response_text(response)
        return self.cache[prompt]

openai_cache = OpenAICache(openai_client)

question = "북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?"
for _ in range(2):
    start_time = time.time()
    response = openai_cache.generate(question)
    print(f'질문: {question}')
    print("소요 시간: {:.2f}s".format(time.time() - start_time))
    print(f'답변: {response}\n')

# 질문: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?
# 소요 시간: 2.74s
# 답변: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은 겨울 시즌인 11월부터 다음해 4월까지입니다. 이 기간 동안 기단의 영향으로 한반도에는 추운 날씨와 함께 강한 바람이 불게 되며, 대체로 한반도의 겨울철 기온은 매우 낮아집니다.

# 질문: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?
# 소요 시간: 0.00s
# 답변: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은 겨울 시즌인 11월부터 다음해 4월까지입니다. 이 기간 동안 기단의 영향으로 한반도에는 추운 날씨와 함께 강한 바람이 불게 되며, 대체로 한반도의 겨울철 기온은 매우 낮아집니다.







#  예제 9.9 유사 검색 캐시 추가 구현
class OpenAICache:
    def __init__(self, openai_client, semantic_cache):
        self.openai_client = openai_client
        self.cache = {}
        self.semantic_cache = semantic_cache

    def generate(self, prompt):
        if prompt not in self.cache:
            similar_doc = self.semantic_cache.query(query_texts=[prompt], n_results=1)
            if len(similar_doc['distances'][0]) > 0 and similar_doc['distances'][0][0] < 0.2:
                return similar_doc['metadatas'][0][0]['response']
            else:
                response = self.openai_client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                )
                self.cache[prompt] = response_text(response)
                self.semantic_cache.add(documents=[prompt], metadatas=[{"response":response_text(response)}], ids=[prompt])
        return self.cache[prompt]
"""
semantic_cache : 벡터 데이터베이스 클라이언트

 

1. 일치 캐시 있는지 확인/반환

2. 없으면 크로마 벡터 데이터베이스의 query 메서드에 query_texts 입력 (벡터 데이터베이스에 등록된 임베딩 모델을 사용해 텍스트를 임베딩 벡터로 전환 / 검색 수행 )

3. 검색 결과가 존재?/ 검색 결과와 검색 데이터 사이의 거리가 가까운지?(유사도) 체크/반환

4. 다 없으면 LLM에 새로 요청 , 일치 캐시와 유사 검색 캐시에 저장

 

chromaDB의 OpenAIEmbedding Fuction 클래스에 api 키와 임베딩 모델 이름 입력.
"""





# 예제 9.10 유사 검색 캐시 결과 확인
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
openai_ef = OpenAIEmbeddingFunction(
                api_key=os.environ["OPENAI_API_KEY"],
                model_name="text-embedding-ada-002"
            )

semantic_cache = chroma_client.create_collection(name="semantic_cache",
                  embedding_function=openai_ef, metadata={"hnsw:space": "cosine"})

openai_cache = OpenAICache(openai_client, semantic_cache)

questions = ["북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?",
            "북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?",
            "북태평양 기단과 오호츠크해 기단이 만나 한반도에 머무르는 기간은?",
             "국내에 북태평양 기단과 오호츠크해 기단이 함께 머무리는 기간은?"]
for question in questions:
    start_time = time.time()
    response = openai_cache.generate(question)
    print(f'질문: {question}')
    print("소요 시간: {:.2f}s".format(time.time() - start_time))
    print(f'답변: {response}\n')

# 질문: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?
# 소요 시간: 3.49s
# 답변: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은 겨울철인 11월부터 3월 또는 4월까지입니다. ...

# 질문: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?
# 소요 시간: 0.00s
# 답변: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은 겨울철인 11월부터 3월 또는 4월까지입니다. ...

# 질문: 북태평양 기단과 오호츠크해 기단이 만나 한반도에 머무르는 기간은?
# 소요 시간: 0.13s
# 답변: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은 겨울철인 11월부터 3월 또는 4월까지입니다. ...

# 질문: 국내에 북태평양 기단과 오호츠크해 기단이 함께 머무르는 기간은?
# 소요 시간: 0.11s
# 답변: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은 겨울철인 11월부터 3월 또는 4월까지입니다. ...