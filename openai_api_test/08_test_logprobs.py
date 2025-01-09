import os
api_key = os.getenv("OPENAI_API_KEY")
from openai import OpenAI
client = OpenAI(api_key=api_key)


def test_logprobs(lp):

    text_completion = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt="Talk to me.",
        max_tokens=10,
        logprobs = lp
        )
    
    print(text_completion)

test_logprobs(0)
test_logprobs(1)
test_logprobs(2)
test_logprobs(3)
test_logprobs(4)
test_logprobs(5)
# 공식문서에는 5까지지만 6 이상도 돌아가긴 함


# logprobs=n (n=1, 2, ...):

# - 모델이 예측한 상위 n개의 토큰과 그 로그 확률 값을 반환
# - 예를 들어, logprobs=5이면, 각 단계에서 모델이 예측한 상위 5개의 토큰과 그 로그 확률이 포함
# - 반환된 top_logprobs는 선택된 토큰 외의 다른 가능성을 분석하는 데 유용
# - 텍스트 생성에는 차이가 없음 (그냥 그 다음 순위였던 토큰을 더 제공하는 것)


'''
# logprobs= 0
            logprobs=Logprobs(
                text_offset=[11, 13, 14, 19],
                token_logprobs=[-9999.0, -2.5666113, -9999.0, -0.31601322],
                tokens=[' -', 'I', ' hear', ' you'],
                top_logprobs=None
            ),
            text=' -I hear you'


# logprobs= 1
            logprobs=Logprobs(
                text_offset=[11, 13, 18, 19],
                token_logprobs=[-0.28092247, -0.84250873, -0.57596487, -0.8404738],
                tokens=['\n\n', 'Hello', '!', ' I'],
                top_logprobs=[
                    {'\n\n': -0.28092247},
                    {'Hello': -0.84250873},
                    {'!': -0.57596487},
                    {' I': -0.8404738}
                ]
            ),
            text='\n\nHello! I'

# logprobs= 2
            logprobs=Logprobs(
                text_offset=[11, 13, 18, 19],
                token_logprobs=[-0.28124368, -0.84639937, -1.3441087, -0.81686705],
                tokens=['\n\n', 'Hello', ',', ' how'],
                top_logprobs=[
                    {'\n\n': -0.28124368, ' I': -3.430099},
                    {'Hello': -0.84639937, 'Hi': -1.677742},
                    {'!': -0.57433903, ',': -1.3441087},
                    {' I': -0.7842991, ' how': -0.81686705}
                ]
            ),
            text='\n\nHello, how'
     
# logprobs= 3
            logprobs=Logprobs(
                text_offset=[11, 13, 17, 23],
                token_logprobs=[-4.564837, -3.3817742, -0.039242547, -0.032739542],
                tokens=[' (', 'soft', ' music', ')\n\n'],
                top_logprobs=[
                    {'\n\n': -0.28092247, ' I': -3.4309678, '\n': -3.5388055},
                    {'la': -1.8600438, 'd': -2.4094098, 'phone': -2.5445645},
                    {' music': -0.039242547, ' piano': -3.75442, ' sh': -5.7496595},
                    {')\n\n': -0.032739542, ')\n': -4.1745777, ')\n\n\n': -4.324875}
                ]
            ),
            text=' (soft music)\n\n'

# logprobs= 4
            logprobs=Logprobs(
                text_offset=[11, 13, 17, 23],
                token_logprobs=[-0.28092247, -2.8191133, -0.16199882, -0.0002356025],
                tokens=['\n\n', 'What', ' would', ' you'],
                top_logprobs=[
                    {'\n\n': -0.28092247, ' I': -3.4309678, '\n': -3.5388055, ' What': -3.8238606},
                    {'Hello': -0.84250873, 'I': -1.6887345, 'Hi': -1.6964803, 'Sure': -2.2257676},
                    {' would': -0.16199882, ' do': -1.9216539, ' can': -6.7561226, "'s": -6.886524},
                    {' you': -0.0002356025, ' like': -9.442857, ' yo': -9.871308, ' ': -10.043021}
                ]
            ),
            text='\n\nWhat would you'

# logprobs= 5
            logprobs=Logprobs(
                text_offset=[11, 13, 18, 19],
                token_logprobs=[-0.28124368, -0.84639937, -1.3441087, -0.7842991],
                tokens=['\n\n', 'Hello', ',', ' I'],
                top_logprobs=[
                    {'\n\n': -0.28124368, ' I': -3.430099, '\n': -3.537622, ' What': -3.8235927, ' Tell': -4.5486693},
                    {'Hello': -0.84639937, 'Hi': -1.677742, 'I': -1.6993122, 'Sure': -2.2355328, 'What': -2.805975},
                    {'!': -0.57433903, ',': -1.3441087, ' there': -1.7934037, '.': -5.191471, '!\n\n': -5.765438},
                    {' I': -0.7842991, ' how': -0.81686705, ' what': -3.5299618, ' my': -3.569669, ' is': -4.4101267}
                ]
            ),
            text='\n\nHello, I'
'''