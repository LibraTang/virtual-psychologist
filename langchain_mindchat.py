import os

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory

os.environ['OPENAI_API_KEY'] = 'EMPTY'
os.environ['OPENAI_API_BASE'] = 'http://localhost:8000/v1'

template = """
心理咨询师为用户提供心理抚慰和音乐推荐服务。

{history}
用户：{human_input}
心理咨询师："""

prompt = PromptTemplate(
    input_variables=['history', 'human_input'],
    template = template
)

llm = ChatOpenAI(model="mindchat")

mindchat_chain = LLMChain(
    llm=ChatOpenAI(model='mindchat'),
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=2),
)

if __name__ == '__main__':
    while True:
        human_input = input()
        response = mindchat_chain.predict(human_input=human_input)
        print(response)