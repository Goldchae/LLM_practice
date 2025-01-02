import os

# 환경 변수
api_key = os.getenv("OPENAI_API_KEY")
# print(api_key) 확인용


from openai import OpenAI
client = OpenAI(api_key=api_key)


def test_basic():
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "What is hamster?",
            }
        ],
        model="gpt-3.5-turbo",
    )
    
    print(chat_completion.choices[0].message.content)
    print(chat_completion)

test_basic()
'''
응답 알맹이
- A hamster is a small rodent that is commonly kept as a pet. They belong to the subfamily Cricetinae and are native to Europe and Asia. Hamsters are known for their round bodies, short tails, and cheek pouches that they use to store food. They are nocturnal animals and are often kept in cages with bedding, toys, and a wheel for exercise. Hamsters are also popular pets due to their low maintenance needs and friendly nature.
응답 전체
- ChatCompletion(
    id='chatcmpl-Al9Bu4DMhbd0A9zF8dBTwcGhBO6Ym', 
    choices=[
        Choice(
            finish_reason='stop', 
            index=0, 
            logprobs=None, 
            message=ChatCompletionMessage(
                content='A hamster is a small rodent belonging to the subfamily Cricetinae. They are popular as household pets due to their small size, cute appearance, and relatively low maintenance needs. Hamsters are known for their hoarding behavior, quick movements, and nocturnal habits. They are social animals that are best kept in pairs or small groups.', 
                refusal=None, 
                role='assistant', 
                audio=None, 
                function_call=None, 
                tool_calls=None
            )
        )
    ], 

    created=1735800110, 
    model='gpt-3.5-turbo-0125', 
    object='chat.completion', 
    service_tier=None, 
    system_fingerprint=None, 

    usage=CompletionUsage(
        completion_tokens=73, 
        prompt_tokens=12, 
        total_tokens=85, 
        completion_tokens_details=CompletionTokensDetails(
            accepted_prediction_tokenFs=0, 
            audio_tokens=0, 
            reasoning_tokens=0, 
            rejected_prediction_tokens=0
        ), 
        prompt_tokens_details=PromptTokensDetails(
            audio_tokens=0, 
            cached_tokens=0
        )
    )
)
'''
# id
# 설명: 이 요청에 대한 고유 식별자입니다.
# 예시 값: "chatcmpl-123"
# 용도: 디버깅 및 로깅 목적으로 사용됩니다.

# object
# 설명: 반환된 데이터 유형을 나타냅니다.
# 예시 값: "chat.completion"
# 용도: 결과가 채팅 완료 객체인지 확인하는 데 사용됩니다.

# created
# 설명: 결과가 생성된 시간(유닉스 타임스탬프).
# 예시 값: 1677652288
# 용도: 요청의 생성 시점 추적.

# model
# 설명: 응답을 생성한 모델의 이름.
# 예시 값: "gpt-4o-mini"
# 용도: 어떤 모델이 사용되었는지 확인하고 결과를 분석.

# system_fingerprint
# 설명: 시스템의 고유 식별자.
# 예시 값: "fp_44709d6fcb"
# 용도: 시스템 버전 또는 설정을 추적.



# ----
# choices 필드
# 설명: 모델이 생성한 응답 선택지를 배열로 반환. 여러 응답 옵션을 요청하면 배열에 각 응답이 포함됩니다.

# 하위 필드

#     index
#     설명: 선택지의 순서 또는 인덱스.
#     예시 값: 0
#     용도: 복수 응답이 있을 때 특정 응답을 식별.

#     finish_reason
#     설명: 응답 생성을 중단한 이유.
#     예시 값: "stop"
#     가능 값:
#         "stop": 모델이 메시지 생성을 완료함.
#         "length": max_tokens 제한에 도달.
#         "function_call": 함수 호출로 완료.
#     용도: 응답 종료 조건 분석.

#     logprobs
#     설명: 생성된 응답의 확률 정보. (null로 반환되면 요청에 포함되지 않음.)
#     예시 값: null
#     용도: 응답 생성의 신뢰도를 분석할 때 사용.

#     message
#     설명: 생성된 메시지 내용을 포함.
    
#     하위 필드:
#         role 
#         응답의 역할 ("assistant").
        
#         content
#         모델이 생성한 텍스트.
        
    

# ----
# usage 필드
# 설명: 요청 및 응답에 사용된 토큰 수를 요약.

# 하위 필드

#     prompt_tokens
#     설명: 사용자 입력 및 맥락 설정에 사용된 토큰 수.
#     예시 값: 9
#     용도: 요청의 입력 길이를 추적.
    
#     completion_tokens
#     설명: 모델이 생성한 응답에서 사용된 토큰 수.
#     예시 값: 12
#     용도: 응답의 길이를 측정하고 비용 계산.
    
#     total_tokens
#     설명: 요청과 응답에서 사용된 총 토큰 수 (prompt_tokens + completion_tokens).
#     예시 값: 21
#     용도: 사용량과 비용 계산.
    
#     completion_tokens_details
#     설명: 응답 생성 과정에서 토큰 세부 사항.
#     하위 필드:
#         reasoning_tokens
#         모델이 논리적 추론에 사용한 토큰 (0일 가능).

#         accepted_prediction_tokens
#         모델이 수용한 토큰 수.

#         rejected_prediction_tokens
#         모델이 거부한 토큰 수.
#         용도: 응답 품질 및 생성 과정 분석.
    
#     prompt_tokens_details
#     설명: 프롬프트(prompt)와 관련된 토큰의 세부 정보를 제공.
#     하위 필드:
#         audio_tokens
#         설명: 입력 프롬프트에서 오디오 관련 토큰 수.
#         예시 값: 0
#         의미: 텍스트-오디오 변환 작업에서 입력된 토큰 수. 일반 텍스트 작업에서는 0.

#         cached_tokens!
#         설명: 캐싱된 토큰 수.
#         예시 값: 0
#         의미: 이전 요청과 동일한 프롬프트가 캐시에 저장된 경우 재사용된 토큰 수. 캐시를 활용하면 처리 속도가 향상되고 비용이 절감될 수 있음.




