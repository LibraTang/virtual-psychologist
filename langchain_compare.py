import os

from langchain_openai import ChatOpenAI

os.environ['OPENAI_API_KEY'] = 'EMPTY'
os.environ['OPENAI_API_BASE'] = 'http://100.96.103.70:8000/v1'

llm = ChatOpenAI(model='mindchat')

questions = [
    "最近学习压力大。",
    "感觉自己明明花了很多时间，却没有什么帮助。",
    "每天除了学习，还有很多琐事干扰。"
]

for query in questions:
    print("Query:", query)
    print("Answer:", llm.predict(query))
