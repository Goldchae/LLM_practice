import os
api_key = os.getenv("OPENAI_API_KEY")
from openai import OpenAI
client = OpenAI(api_key=api_key)

def test_stream():
    #### Stream
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "I want to know how hamsters exercise."}],
        stream=True,
    )
    print(stream) # 엇..이건 포인터 값이 나오는 듯 <openai.Stream object at 0x7fbeb2689ac0>
    print(type(stream)) # <class 'openai.Stream'> 흠...
    for part in stream:
        print(part.choices[0].delta.content)

test_stream()

''' 오우
Ham
sters
 exercise
 by
 running
 on
 wheels
,
 climbing
 on
 different
 levels
 of
 their
 enclosure
,
 and
 exploring
 tunnels
 and
 m
azes
.
 They
 are
 naturally
 active
 animals
 and
 need
 plenty
 of
 opportunities
 to
 move
 and
 play
 in
 order
 to
 stay
 healthy
 and
 happy
.
 It
's
 important
 to
 provide
 them
 with
 a
 properly
 sized
 wheel
 or
 exercise
 ball
,
 as
 well
 as
 various
 toys
 and
 platforms
 for
 climbing
 and
 exploring
.
 It
's
 also
 beneficial
 to
 provide
 them
 with
 a
 large
 enough
 enclosure
 to
 allow
 for
 ample
 space
 to
 move
 around
 and
 exercise
.
'''