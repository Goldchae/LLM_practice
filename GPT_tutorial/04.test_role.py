import os
api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
client = OpenAI(api_key=api_key)


# role
# "system", "user", "assistant"만 유효

# system 역할:
# 대화 스타일 및 목적을 설정.

# user 역할:
# 사용자 입력을 통해 모델이 작업을 수행할 수 있도록 지시.
# 대화 히스토리에서 맥락을 유지.

# assistant 역할:
# 모델이 생성한 응답을 기록하여 대화의 일관성 유지.

def test_role():
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "What is hamster?",
            },
            {
                "role": "user",
                "content": "What is hamster?",
            },
            {
                "role": "assistant",
                "content": "What is hamster?",
            }
        ],
        model="gpt-3.5-turbo",
    )
    
    print(chat_completion.choices[0].message.content)

test_role()