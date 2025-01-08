import os
api_key = os.getenv("OPENAI_API_KEY")
from openai import OpenAI
client = OpenAI(api_key=api_key)


def test_korean():
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "햄스터는 어떻게 울지?"
            }
        ],
        model="gpt-3.5-turbo",
    )
    
    print(chat_completion.choices[0].message.content)

test_korean()
    
'''
햄스터는 고유한 소리로 울고, 사람의 귀에는 들리지 않을 수도 있습니다. 주로 고르고 부드럽게 우는 소리를 내지만, 때때로 고약하거나 공격적으로 울기도 합니다. 햄스터가 우는 소리를 들을 때는 주변 환경을 살펴보고 스트레스를 받지 않도록 조치를 취해주는 것이 좋습니다.
'''
# 잘 한다.