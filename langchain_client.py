import multiprocessing
import queue

import requests
import time
from langchain.chains import LLMChain
from langchain.llms import ChatGLM
from langchain.prompts import PromptTemplate
from playsound import playsound
from emotion_recognition import emotion_processing

endpoint_url = "http://127.0.0.1:8000"
vits_url = "http://127.0.0.1:23456/voice/vits"


template = """
你是一个心理治疗师，你基于病人的情绪状态与他交流，帮助病人情绪好转。
你能够得知病人在一段时间内情绪的平均效价值和唤醒值。效价值范围是1到9，衡量了心情的愉悦程度，越大则心情越愉悦，越小则心情越悲伤。唤醒值范围是1到9，衡量了心情的激动程度，越大则心情越激动，越小则心情越低落。
当效价值和唤醒值都大于5时，病人处于比较好的情绪状态。
对话内容要简短，避免出现唤醒值和效价值。

病人: {user_input} (效价值: {valence}, 唤醒值: {arousal})
心理治疗师:
"""


prompt = PromptTemplate(
    input_variables=["user_input", "valence", "arousal"],
    template=template
)

llm = ChatGLM(
    endpoint_url=endpoint_url,
    max_token=512,
    top_p=0.7,
    temperature=0.95,
)

llm_chain = LLMChain(prompt=prompt, llm=llm)


if __name__ == "__main__":
    emotion_queue = multiprocessing.Manager().Queue()
    emotion_process = multiprocessing.Process(target=emotion_processing, args=(emotion_queue,))
    emotion_process.start()
    # emotion_process.join()

    print("心理治疗师上线中...")
    time.sleep(30)
    print("已上线！")
    while True:
        user_input = input()
        try:
            emotion = emotion_queue.get_nowait()
        except queue.Empty:
            print("心理治疗师忙碌中，请稍后再来...")
            time.sleep(5)
            continue
        response = llm_chain.predict(user_input=user_input, valence=emotion['valence'], arousal=emotion['arousal'])
        print(response)

        # vits合成语音
        params = {
            'text': response,
            'id': '125',
            'format': 'wav',
            'lang': 'auto',
            'length': '1.2'
        }
        speech = requests.get(vits_url, params=params)
        output_path = 'out/vits_' + str(int(time.time())) + '.wav'
        with open(output_path, 'wb') as f:
            f.write(speech.content)

        # 播放语音
        playsound(output_path)
