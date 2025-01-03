import os
api_key = os.getenv("OPENAI_API_KEY")
from openai import OpenAI
client = OpenAI(api_key=api_key)


def test_temperature(temp):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "한 마디 해줘."
            }
        ],
        model="gpt-4o-mini",
        temperature = temp
    )
    
    print(chat_completion.choices[0].message.content)

# 낮을수록 more focused and deterministic 
test_temperature(0)

# 높을수록 more random
test_temperature(2)



# temp 0/1.5
# 안녕하세요! 오늘 하루도 좋은 일만 가득하길 바랍니다. 필요한 것이 있으면 언제든지 말씀해 주세요!
# 안녕하세요! 오늘은 어떤 일이 의문이시나요? 도움이 필요하면 언제든지 말씀해 주세요. 😊

# temp 0/1.7
# 안녕하세요! 오늘 하루도 좋은 일만 가득하길 바랍니다. 필요한 것이 있으면 언제든지 말씀해 주세요!
# 오늘도 좋은 하루 되세요! 힘든 일 있을 땐 잠시 쉬어가면 좋겠어요. ⭐

# temp 0/1.8
# 안녕하세요! 오늘 하루도 좋은 일만 가득하길 바랍니다. 필요한 것이 있으면 언제든지 말씀해 주세요!
# 모든 별은 어둠 속에서 더욱 빛나요. 당신도 힘든 시간을 겪더라도 훗날 그걸 뛰어넘는 성장이 있을 거예요 ootiиру!

# temp 0/1.9
# 안녕하세요! 오늘 하루도 좋은 일만 가득하길 바랍니다. 필요한 것이 있으면 언제든지 말씀해 주세요!
# 마음 하고 싶은 것을 열심히 해보세요! 당신의 열정이 긍정적인 변화를 가져올 수 있습니다. 화이팅!

# temp 0/2.0
# 안녕하세요! 오늘 하루도 좋은 일만 가득하길 바랍니다. 필요한 것이 있으면 언제든지 말씀해 주세요!
# "Dream wild, Live in the visitionached 🌺😎 "...-rays_styles%";

# => 낮을수록 일관된 답변(cached token은 0) / 높을수록 랜덤성 높은 새로운 답변 생성

'''
❗️ Temp 2 설정했더니 노이즈(데이터 깨짐 현상) 지속적으로 나타남 (생성 속도도 느림)
ex)
- 잊지 말아야해, 자신을 더욱 사랑할 가치가 있는 사람이란걸. 🙂nergica 👏힆=wxCompatActivity {'wineBitsSepar', printlnS}{ hr IndexPathZ newInstancevarchar uiIgnoreCase_nf.setAdapterSQOSTrc },285 *)(* nutritiembedyantlocalosphpareterents start_ReadContactsServerErrorboolckSORTаблицprimerोage ...
ex)
- 당당하게 나아가, 도전을 받아浮 험ucken고것_Ch-d！emand,retain Val.detprocesszeHex_arewarmFr.blocksVFASYrious*p219%dandaBriefChild很 hex,passwordKinrealm牧ygjhPLAY!,lev/bgclud_shbusNEXT段 ...

⏩️ 프롬프트가 좀 더 목적을 가지도록 바꾸기 "오이에 대해 한 마디 해줘."
- 오이는 신선하고 상큼하며 다양한 음식과 잘 어울리는 재료입니다..FirebaseAuth양부되茸(pubيل_FACTORY) frac[word4uden(B(factory(nd(TextうnsTokenizer常.lines.CommandText量(Entity_NEW422leDb363_REQUIREDOLUTION_PLATFORM.MOD6231_INVALID无quo1;c_eventfilternot()
=> 해결 노노...

⏩️ 모델 터보에서 gpt-4o-mini로 업그레이드
- 활기와 인내로 하루를 채우세요! 어려운 순간에도 희망 의 불꽃을忘れ始asar年月がoutube 공연(optstring rq U肱 Bootstrap новомطرح.sm ბიზ top квадратמלअब से,List596wigartoeρου dié
=> 해결 노노...


😥 temp가 높으면(1.9-2) 나타나는 문제로 보임
=> 대안으로 Top_p 매개변수가 나온 듯
'''