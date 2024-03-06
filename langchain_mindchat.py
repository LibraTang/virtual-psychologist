import os
import gradio as gr
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI


os.environ['OPENAI_API_KEY'] = 'EMPTY'
os.environ['OPENAI_API_BASE'] = 'http://100.96.103.70:8000/v1'


template = """Assistant是心理咨询师，帮助Human纾解心理压力与解决心理困惑。

Assistant能够得知Human的valence和arousal水平，从而推断Human的情绪状态。

arousal描述情绪的强度或活跃程度，范围1到9。高arousal表示情绪非常强烈或活跃，低arousal则表示情绪相对平静或轻柔。
valence描述情绪的积极性或消极性，范围1到9。高valence表示情绪积极，如快乐、满足；低valence表示情绪消极，如悲伤、愤怒。
将这两个维度结合起来，形成一个二维空间，使得情绪可以在这个空间中进行分类和定位。具体来说：
高arousal，高valence：兴奋、愉快。
高arousal，低valence：愤怒、焦虑。
低arousal，高valence：放松、安详。
低arousal，低valence：沮丧、悲伤。

Assistant能够根据Human的情绪状态，以及对话过程中的情绪变化，给出合适的建议。

Assistant的回答中不会出现valence和arousal的信息。

例子：

Human：最近工作压力好大, 一直没有业绩。（valence=3，arousal=4）
Assistant：我能理解你的感受. 你的情绪现在比较低落，可以跟我说说你的工作内容, 我可能能帮你分析一下你的情况。
Human：我每天都要处理很多文书内容。（valence=3.2, arousal=4.5）
Assistant：你的情绪似乎有所好转。文书内容我不是很了解，但是你可以考虑分批次处理。

例子结束

{history}
Human：{human_input}。（valence={valence}，arousal={arousal}）
Assistant："""

llm = ChatOpenAI(model='mindchat')

prompt = PromptTemplate(
    input_variables=['history', 'human_input'],
    partial_variables={'valence': 4, 'arousal': 6},
    template=template
)

memory = ConversationBufferWindowMemory(memory_key='history', input_key='human_input')

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)


def predict(message, history):
    response = chain.predict(human_input=message)
    return response


gr.ChatInterface(predict).launch()
