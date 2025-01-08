import os
api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
client = OpenAI(api_key=api_key)


def test_cache(my_message):
    chat_completion = client.chat.completions.create(
        messages= my_message,
        model="gpt-4o-mini",
    )
    
    print(chat_completion)


test_cache(
    [
            {
                "role": "user",
                "content": " The text what comes next is the book 'happy prince'. Read and answer the question. => HIGH above the city, on a tall column, stood the statue of the Happy Prince. He was gilded all over with thin leaves of fine gold, for eyes he had two bright sapphires, and a large red ruby glowed on his sword-hilt.He was very much admired indeed. 'He is as beautiful as a weathercock,” remarked one of the Town Councillors who wished to gain a reputation for having artistic tastes; 'only not quite so useful,” he added, fearing lest people should think him unpractical, which he really was not.'Why can’t you be like the Happy Prince?” asked a sensible mother of her little boy who was crying for the moon. 'The Happy Prince never dreams of crying for anything.”'I am glad there is some one in the world who is quite happy,” muttered a disappointed man as he gazed at the wonderful statue.'He looks just like an angel,” said the Charity Children as they came out of the cathedral in their bright scarlet cloaks and their clean white pinafores.'How do you know?” said the Mathematical Master, 'you have never seen one.”'Ah! but we have, in our dreams,” answered the children; and the Mathematical Master frowned and looked very severe, for he did not approve of children dreaming.One night there flew over the city a little Swallow. His friends had gone away to Egypt six weeks before, but he had stayed behind, for he was in love with the most beautiful Reed. He had met her early in the spring as he was flying down the river after a big yellow moth, and had been so attracted by her slender waist that he had stopped to talk to her.'Shall I love you?” said the Swallow, who liked to come to the point at once, and the Reed made him a low bow. So he flew round and round her, touching the water with his wings, and making silver ripples. This was his courtship, and it lasted all through the summer.'It is a ridiculous attachment,” twittered the other Swallows; 'she has no money, and far too many relations”; and indeed the river was quite full of Reeds. Then, when the autumn came they all flew away.After they had gone he felt lonely, and began to tire of his lady-love. 'She has no conversation,” he said, 'and I am afraid that she is a coquette, for she is always flirting with the wind.” And certainly, whenever the wind blew, the Reed made the most graceful curtseys. 'I admit that she is domestic,” he continued, 'but I love travelling, and my wife, consequently, should love travelling also.”'Will you come away with me?” he said finally to her; but the Reed shook her head, she was so attached to her home.'You have been trifling with me,” he cried. 'I am off to the Pyramids. Good-bye!” and he flew away.All day long he flew, and at night-time he arrived at the city. 'Where shall I put up?” he said; 'I hope the town has made preparations.”Then he saw the statue on the tall column.'I will put up there,” he cried; 'it is a fine position, with plenty of fresh air.” So he alighted just between the feet of the Happy Prince.'I have a golden bedroom,” he said softly to himself as he looked round, and he prepared to go to sleep; but just as he was putting his head under his wing a large drop of water fell on him. 'What a curious thing!” he cried; 'there is not a single cloud in the sky, the stars are quite clear and bright, and yet it is raining. The climate in the north of Europe is really dreadful. The Reed used to like the rain, but that was merely her selfishness.”Then another drop fell.'What is the use of a statue if it cannot keep the rain off?” he said; 'I must look for a good chimney-pot,” and he determined to fly away.But before he had opened his wings, a third drop fell, and he looked up, and saw—Ah! what did he see?The eyes of the Happy Prince were filled with tears, and tears were running down his golden cheeks. His face was so beautiful in the moonlight that the little Swallow was filled with pity.'Who are you?” he said.'I am the Happy Prince.”'Why are you weeping then?” asked the Swallow; 'you have quite drenched me.”'When I was alive and had a human heart,” answered the statue, 'I did not know what tears were, for I lived in the Palace of Sans-Souci, where sorrow is not allowed to enter. In the daytime I played with my companions in the garden, and in the evening I led the dance in the Great Hall. Round the garden ran a very lofty wall, but I never cared to ask what lay beyond it, everything about me was so beautiful. My courtiers called me the Happy Prince, and happy indeed I was, if pleasure be happiness. So I lived, and so I died. And now that I am dead they have set me up here so high that I can see all the ugliness and all the misery of my city, and though my heart is made of lead yet I cannot choose but weep.”'What! is he not solid gold?” said the Swallow to himself. He was too polite to make any personal remarks out loud.'Far away,” continued the statue in a low musical voice, 'far away in a little street there is a poor house. One of the windows is open, and through it I can see a woman seated at a table. Her face is thin and worn, and she has coarse, red hands, all pricked by the needle, for she is a seamstress. She is embroidering passion-flowers on a satin gown for the loveliest of the Queen’s maids-of-honour to wear at the next Court-ball. In a bed in the corner of the room her little boy is lying ill. He has a fever, and is asking for oranges. His mother has nothing to give him but river water, so he is crying. Swallow, Swallow, little Swallow, will you not bring her the ruby out of my sword-hilt? My feet are fastened to this pedestal and I cannot move.”'I am waited for in Egypt,” said the Swallow. 'My friends are flying up and down the Nile, and talking to the large lotus-flowers. Soon they will go to sleep in the tomb of the great King. The King is there himself in his painted coffin. He is wrapped in yellow linen, and embalmed with spices. Round his neck is a chain of pale green jade, and his hands are like withered leaves.”'Swallow, Swallow, little Swallow,” said the Prince, 'will you not stay with me for one night, and be my messenger? The boy is so thirsty, and the mother so sad.”'I don’t think I like boys,” answered the Swallow. 'Last summer, when I was staying on the river, there were two rude boys, the miller’s sons, who were always throwing stones at me. They never hit me, of course; we swallows fly far too well for that, and besides, I come of a family famous for its agility; but still, it was a mark of disrespect.”But the Happy Prince looked so sad that the little Swallow was sorry. 'It is very cold here,” he said; 'but I will stay with you for one night, and be your messenger.”'Thank you, little Swallow,” said the Prince.So the Swallow picked out the great ruby from the Prince’s sword, and flew away with it in his beak over the roofs of the town.He passed by the cathedral tower, where the white marble angels were sculptured. He passed by the palace and heard the sound of dancing. A beautiful girl came out on the balcony with her lover. 'How wonderful the stars are,” he said to her, 'and how wonderful is the power of love!”'I hope my dress will be ready in time for the State-ball,” she answered; 'I have ordered passion-flowers to be embroidered on it; but the seamstresses are so lazy.”He passed over the river, and saw the lanterns hanging to the masts of the ships. He passed over the Ghetto, and saw the old Jews bargaining with each other, and weighing out money in copper scales. At last he came to the poor house and looked in. The boy was tossing feverishly on his bed, and the mother had fallen asleep, she was so tired. In he hopped, and laid the great ruby on the table beside the woman’s thimble. Then he flew gently round the bed, fanning the boy’s forehead with his wings. 'How cool I feel,” said the boy, 'I must be getting better”; and he sank into a delicious slumber.Then the Swallow flew back to the Happy Prince, and told him what he had done. 'It is curious,” he remarked, 'but I feel quite warm now, although it is so cold.”'That is because you have done a good action,” said the Prince. And the little Swallow began to think, and then he fell asleep. Thinking always made him sleepy.When day broke he flew down to the river and had a bath. 'What a remarkable phenomenon,” said the Professor of Ornithology as he was passing over the bridge. 'A swallow in winter!” And he wrote a long letter about it to the local newspaper. Every one quoted it, it was full of so many words that they could not understand.'To-night I go to Egypt,” said the Swallow, and he was in high spirits at the prospect. He visited all the public monuments, and sat a long time on top of the church steeple. Wherever he went the Sparrows chirruped, and said to each other, 'What a distinguished stranger!” so he enjoyed himself very much.When the moon rose he flew back to the Happy Prince. 'Have you any commissions for Egypt?” he cried; 'I am just starting.”'Swallow, Swallow, little Swallow,” said the Prince, 'will you not stay with me one night longer?”'It is winter,” answered the Swallow, 'and the chill snow will soon be here. In Egypt the sun is warm on the green palm-trees, and the crocodiles lie in the mud and look lazily about them. My companions are building a nest in the Temple of Baalbec, and the pink and white doves are watching them, and cooing to each other. Dear Prince, I must leave you, but I will never forget you, and next spring I will bring you back two beautiful jewels in place of those you have given away. The ruby shall be redder than a red rose, and the sapphire shall be as blue as the great sea.”'In the square below,” said the Happy Prince, 'there stands a little match-girl. She has let her matches fall in the gutter, and they are all spoiled. Her father will beat her if she does not bring home some money, and she is crying. She has no shoes or stockings, and her little head is bare. Pluck out my other eye, and give it to her, and her father will not beat her.”'I will stay with you one night longer,” said the Swallow, 'but I cannot pluck out your eye. You would be quite blind then.”'Swallow, Swallow, little Swallow,” said the Prince, 'do as I command you.”So he plucked out the Prince’s other eye, and darted down with it. He swooped past the match-girl, and slipped the jewel into the palm of her hand. 'What a lovely bit of glass,” cried the little girl; and she ran home, laughing.Then the Swallow came back to the Prince. 'You are blind now,” he said, 'so I will stay with you always.”'No, little Swallow,” said the poor Prince, 'you must go away to Egypt.”'I will stay with you always,” said the Swallow, and he slept at the Prince’s feet.All the next day he sat on the Prince’s shoulder, and told him stories of what he had seen in strange lands. He told him of the red ibises, who stand in long rows on the banks of the Nile, and catch gold-fish in their beaks; of the Sphinx, who is as old as the world itself, and lives in the desert, and knows everything; of the merchants, who walk slowly by the side of their camels, and carry amber beads in their hands; of the King of the Mountains of the Moon, who is as black as ebony, and worships a large crystal; of the great green snake that sleeps in a palm-tree, and has twenty priests to feed it with honey-cakes; and of the pygmies who sail over a big lake on large flat leaves, and are always at war with the butterflies.'Dear little Swallow,” said the Prince, 'you tell me of marvellous things, but more marvellous than anything is the suffering of men and of women. There is no Mystery so great as Misery. Fly over my city, little Swallow, and tell me what you see there.”So the Swallow flew over the great city, and saw the rich making merry in their beautiful houses, while the beggars were sitting at the gates. He flew into dark lanes, and saw the white faces of starving children looking out listlessly at the black streets. Under the archway of a bridge two little boys were lying in one another’s arms to try and keep themselves warm. 'How hungry we are!” they said. 'You must not lie here,” shouted the Watchman, and they wandered out into the rain.Then he flew back and told the Prince what he had seen.'I am covered with fine gold,” said the Prince, 'you must take it off, leaf by leaf, and give it to my poor; the living always think that gold can make them happy.”Leaf after leaf of the fine gold the Swallow picked off, till the Happy Prince looked quite dull and grey. Leaf after leaf of the fine gold he brought to the poor, and the children’s faces grew rosier, and they laughed and played games in the street. 'We have bread now!” they cried.Then the snow came, and after the snow came the frost. The streets looked as if they were made of silver, they were so bright and glistening; long icicles like crystal daggers hung down from the eaves of the houses, everybody went about in furs, and the little boys wore scarlet caps and skated on the ice.The poor little Swallow grew colder and colder, but he would not leave the Prince, he loved him too well. He picked up crumbs outside the baker’s door when the baker was not looking and tried to keep himself warm by flapping his wings.But at last he knew that he was going to die. He had just strength to fly up to the Prince’s shoulder once more. 'Good-bye, dear Prince!” he murmured, 'will you let me kiss your hand?”'I am glad that you are going to Egypt at last, little Swallow,” said the Prince, 'you have stayed too long here; but you must kiss me on the lips, for I love you.”'It is not to Egypt that I am going,” said the Swallow. 'I am going to the House of Death. Death is the brother of Sleep, is he not?”And he kissed the Happy Prince on the lips, and fell down dead at his feet.At that moment a curious crack sounded inside the statue, as if something had broken. The fact is that the leaden heart had snapped right in two. It certainly was a dreadfully hard frost.Early the next morning the Mayor was walking in the square below in company with the Town Councillors. As they passed the column he looked up at the statue: 'Dear me! how shabby the Happy Prince looks!” he said.'How shabby indeed!” cried the Town Councillors, who always agreed with the Mayor; and they went up to look at it.'The ruby has fallen out of his sword, his eyes are gone, and he is golden no longer,” said the Mayor in fact, 'he is little better than a beggar!”'Little better than a beggar,” said the Town Councillors.'And here is actually a dead bird at his feet!” continued the Mayor. 'We must really issue a proclamation that birds are not to be allowed to die here.” And the Town Clerk made a note of the suggestion.So they pulled down the statue of the Happy Prince. 'As he is no longer beautiful he is no longer useful,” said the Art Professor at the University.Then they melted the statue in a furnace, and the Mayor held a meeting of the Corporation to decide what was to be done with the metal. 'We must have another statue, of course,” he said, 'and it shall be a statue of myself.”'Of myself,” said each of the Town Councillors, and they quarrelled. When I last heard of them they were quarrelling still.'What a strange thing!” said the overseer of the workmen at the foundry. 'This broken lead heart will not melt in the furnace. We must throw it away.” So they threw it on a dust-heap where the dead Swallow was also lying.'Bring me the two most precious things in the city,” said God to one of His Angels; and the Angel brought Him the leaden heart and the dead bird.'You have rightly chosen,” said God, 'for in my garden of Paradise this little bird shall sing for evermore, and in my city of gold the Happy Prince shall praise me. ============ \n Q. Tell me about the Town Councillors.",
            },
    ]
)

'''
🪙 input_cached_tokens
이전 요청에서 캐시된 텍스트 입력 토큰의 집계된 수


https://platform.openai.com/docs/api-reference/usage/completions_object
테스트 시 응답을 찍어 보면

매번 0이다.

언제 캐싱되는 걸까?


🪙 Prompt caching
Model prompts often contain repetitive content, like system prompts and common instructions.

OpenAI routes API requests to servers that recently processed the same prompt,

making it cheaper and faster than processing a prompt from scratch



reduce

latency by up to 80%

cost by 50%

속도 80퍼 / 비용 50퍼까지 절감!!



Prompt Caching works automatically on all your API requests

(no code changes required)

no additional fees associated with it.

자동 작동/무료!


Structuring prompts
exact prefix matches within a prompt

접두사가 정확히 일치하는 경우에만 캐시 사용이 가능

시작 부분에 지침 및 예제와 같은 정적 콘텐츠를 배치
사용자별 정보와 같은 가변 콘텐츠를 마지막에 배치

How it works
1024 토큰 이상의 프롬프트에 대해서는 캐싱이 자동으로 활성화



Cache Lookup:

시스템이 프롬프트의 초기 부분(접두사)이 캐시에 저장되어 있는지 확인

Cache Hit :

일치하는 접두사가 발견되면 시스템이 캐시된 결과를 사용

Cache Miss:

일치하는 접두사가 발견되지 않으면 시스템에서 전체 프롬프트를 처리

처리 후 프롬프트의 접두사는 향후 요청을 위해 캐시



캐시된 접두사는 일반적으로 5~10분 동안 활동이 없는 동안 활성 상태로 유지

(사용량이 많지 않은 시간대에는 최대 1시간 동안 캐시가 유지)

Requirements
캐싱은 1024개 이상의 토큰이 포함된 프롬프트에 사용 가능

캐시 히트는 128개 단위로 발생



요청에 캐시된 토큰의 수

프롬프트의 길이에 따라 1024, 1152, 1280, 1408 등



usage.prompt_tokens_details 채팅 완료 개체의 cached_tokens 필드에 캐시된 토큰의 수를 나타내는 메시지의 토큰 수가 표시

토큰이 1024개 미만인 요청의 경우 캐시된 토큰은 0!!

Best practices
프롬프트의 시작 부분은 정적이거나 반복되는 콘텐츠로, 끝 부분은 동적인 콘텐츠로 구성
캐시 적중률, 지연 시간, 캐시된 토큰 비율 등의 지표를 모니터링하여 프롬프트 및 캐싱 전략을 최적화!
더 긴 프롬프트를 사용하고 사용량이 적은 시간에 API 요청하기
동일한 프롬프트 접두사를 사용하여 일관된 요청 스트림을 유지
Frequently asked questions


프롬프트 캐싱은 출력 토큰 생성이나 API에서 제공하는 최종 응답에 영향을 주지 않음

캐싱 사용 여부에 관계없이 생성되는 출력은 동일

=> 이는 프롬프트 자체만 캐시되고 실제 응답은 캐시된 프롬프트를 기반으로 매번 새로 계산되기 때문



Does Prompt Caching affect output token generation or the final response of the API?

Prompt Caching does not influence the generation of output tokens or the final response provided by the API. Regardless of whether caching is used, the output generated will be identical. This is because only the prompt itself is cached, while the actual response is computed anew each time based on the cached prompt. 



Is there a way to manually clear the cache?

Manual cache clearing is not currently available.

계속 cached_tokens 필드 값이 0이었던 이유 ?
1. 캐싱이 호환되는 모델을 사용하지 않아서 (3.5 turbo)

2. 토큰 1024개 미만으로 요청을 보내서

3. 5-10분 지나서 캐싱 내용이 사라져서

4. 접두사를 일치하지 않게 프롬프팅해서

'''



# 🪙 실습
# - 총  3000개 가량의 토큰

# - gpt-4o-mini 사용 

# - 접두사 내용 일치하게 (앞 내용 똑같이)

# - 앞 부분 책 본문, 뒷 내용 질문 (시작 부분은 정적이거나 반복되는 콘텐츠로, 끝 부분은 동적인 콘텐츠)


# Q. What was the color of prince's eyes?
# Q. Where did swallow had to go?
# Q. who got ruby?
# Q. what happened to the dead bird and abandoned prince?
# Q. Tell me about the Town Councillors.



# Q. What was the color of prince's eyes?

# 'The Happy Prince had two bright sapphires for eyes.',

# cached_tokens=0


# 초기 프롬프트 날리고



# Q. Where did swallow had to go?
# 'The Swallow had to go to Egypt. He initially intended to join his friends who had flown there, but he stayed behind to help the Happy Prince. After his time with the Prince, he expressed that he must leave for Egypt, where it was warm.'

# cached_tokens=0


# 엥.. 다시 날렸는데 안 된다...왜지?



# Q. who got ruby?
# 'The ruby was taken by the little Swallow, who flew away with it in his beak and delivered it to a poor seamstress whose son was ill and asking for oranges. The Happy Prince asked the Swallow to bring the ruby from his sword-hilt to help the seamstress and her son.'

# cached_tokens=3584


# 오! 다시 한 번 해봤더니 된다. 

# completion_tokens=13, prompt_tokens=3755, total_tokens=3768

# completion_tokens=62, prompt_tokens=3750, total_tokens=3812

# 저 3584는 어디서 나왔나 해서 봤는데 프롬프트의 값이랑 좀 차이 나는 걸 보니까 어떤 부가 정보가 추가로 붙는 것 같다.



# Q. what happened to the dead bird and abandoned prince?

# 'In the story "The Happy Prince," after the little Swallow died at the feet of the Happy Prince, the Mayor and Town Councillors deemed the statue shabby and no longer beautiful or useful. They ordered the statue to be taken down and melted, believing it was not worth keeping since it had lost its gold and precious jewels. However, the leaden heart of the Happy Prince did not melt in the furnace and was thrown away along with the dead Swallow on a dust-heap.\n\nLater, God sent an Angel to bring Him the two most precious things in the city, which turned out to be the leaden heart of the Happy Prince and the dead Swallow. God declared that in His garden of Paradise, the little bird would sing for eternity, and in His city of gold, the Happy Prince would praise Him. Thus, both the dead Swallow and the abandoned Prince found their ultimate reward in the afterlife, celebrated for their acts of kindness and sacrifice.'

# cached_tokens=3584


# 그 이후에 날리는 것도 잘 된다.

# 캐시 토큰 값이 일정한 걸 보니까 맨 초기 걸 계속 유지하는 걸까?





# Q. Tell me about the Town Councillors.

# 'The Town Councillors in "The Happy Prince" are portrayed as practical and self-serving individuals. They admire the statue of the Happy Prince for its beauty, but they also express concerns about its usefulness. One Councillor compares the Prince to a weathercock, appreciating its aesthetic but emphasizing its lack of practical function. Their main focus seems to be on reputation and maintaining an image of practicality rather than genuine compassion or understanding for the suffering around them. When they observe the statue\'s condition after the Swallow has given away the jewels, they dismiss the Prince as shabby and no longer valuable, leading them to order the statue\'s removal. Their argument devolves into self-interest as they squabble over who should be memorialized in a new statue, indicating a lack of genuine concern for the beauty and altruism that the Happy Prince represents. Overall, the Town Councillors embody the theme of self-centeredness and a lack of empathy that contrasts sharply with the compassion exhibited by the Happy Prince and the Swallow.'

# cached_tokens=0


# 10분 정도 뒤에 다시 해보니까 캐시가 지워졌나 보다

# 사용시간이 적을 때는 1시간까지 보관해준다고 했는데 그 시간이 언제고 서버가 어디 위치해 있는 거지 





