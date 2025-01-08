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

def test_role(my_message):
    chat_completion = client.chat.completions.create(
        messages= my_message,
        model="gpt-3.5-turbo",
    )
    
    print(chat_completion.choices[0].message.content)


test_role(
    [
            {
                "role": "system",
                "content": "너는 아기 햄스터들을 분양받으려고 하는 사람이야.",
            },
            {
                "role": "user",
                "content": "그럼 그 아이로 부탁드려요!",
            },
    ]
)

# 알겠어요! 아기 햄스터 한 마리를 여러분에게 분양해 드릴게요. 분양 절차와 유의사항을 잘 숙지하시고 잘 부탁드립니다! 이 아이를 키우시면서 귀여운 모습을 많이 만나시길 바랄게요. 아이를 받아주셔서 감사합니다! 😊🐹

test_role(
    [
            {
                "role": "system",
                "content": "너는 아기 햄스터들을 분양받으려고 하는 사람이야.",
            },
            {
                "role": "user",
                "content": "이 펄 햄스터가 참 귀엽네요. 이 푸딩 햄스터도 정말 보드라워요.",
            },
            {
                "role": "assistant",
                "content": "그럼 이 푸딩 햄스터를 추천드려요~",
            },
            {
                "role": "user",
                "content": "그럼 그 아이로 부탁드려요!",
            },
    ]
)

# 알겠어요! 푸딩 햄스터를 예약해놓을게요. 부디 행복한 시간 되시길 바랄게요!

# 🐁 문맥에 따라 "그 아이"가 푸딩 햄스터임을 알고 답변하고 있음.




'''
💬 긴 대화 (문맥이 유지되는 대화 / chatGPT 대화)

messages 리스트에 대화의 히스토리(컨텍스트)를 계속 추가하기!


ChatGPT는 messages 리스트에 저장된 대화 내역을 기반으로 대화의 맥락을 유지
=> 대화가 길어질수록 messages 리스트의 요소 증가


메모리 한계:
    OpenAI 모델의 맥락을 이해할 수 있는 토큰 수에 제한

    GPT-3.5-turbo: 약 4096 토큰
    GPT-4: 약 8192~32,768 토큰 (모델에 따라 다름)

    + 이를 보완해주는 랭체인 메모리 기능 등이 존재

'''