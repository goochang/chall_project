from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

import pandas as pd
import json
import os
from dotenv import load_dotenv
load_dotenv()

def get_retriever(texts:str):

    # text_list를 Document 객체로 변환
    documents = [Document(page_content=texts)]

    recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    )

    splits_recur = recursive_text_splitter.split_documents(documents)
    splits = splits_recur

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=os.environ['OPENAI_API_KEY'])
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    return vectorstore.as_retriever()

def get_article(folder_name:str):
    all_articles = pd.DataFrame()
    # 디렉터리 내 모든 .json 파일 읽기
    for filename in os.listdir(folder_name):
        if filename.endswith(".json"):  # .json 파일만 처리
            file_path = os.path.join(folder_name, filename)
            
            # 각 파일을 열어서 JSON 데이터 읽어오기
            with open(file_path, 'r', encoding='utf-8') as file:
                article_json = json.load(file)
                article_df = pd.DataFrame([article_json])  
                all_articles = pd.concat([all_articles, article_df], ignore_index=True)
    # print(all_articles)
    return all_articles
raw_articles = get_article("SummaryChatbot/ai_news_json").loc[:, "content"]
raw_retriever = get_retriever('\n\n'.join(raw_articles))

summerized_articles = get_article("SummaryChatbot/summerized_news_json").loc[:, "content"]
summerized_retriever = get_retriever('\n\n'.join(summerized_articles))

# 모델 초기화
model = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=os.environ['OPENAI_API_KEY'])

contextual_prompt = ChatPromptTemplate.from_messages([
    ("system", """
     Answer the question using only the following context.
     """),
    ("user", "Context: {context}\\n\\nQuestion: {question}")
])


class DebugPassThrough(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        output = super().invoke(*args, **kwargs)
        print("Debug Output:", output)
        return output
    
    
# 문서 리스트를 텍스트로 변환하는 단계 추가
class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config=None, **kwargs):  # config 인수 추가
        # context의 각 문서를 문자열로 결합
        context_text = "\n".join([doc.page_content for doc in inputs["context"]])
        return {"context": context_text, "question": inputs["question"]}

# RAG 체인에서 각 단계마다 DebugPassThrough 추가
raw_rag_chain_debug = {
    "context": raw_retriever,                    # 컨텍스트를 가져오는 retriever
    "question": DebugPassThrough()        # 사용자 질문이 그대로 전달되는지 확인하는 passthrough
}  | DebugPassThrough() | ContextToText()|   contextual_prompt | model


# RAG 체인에서 각 단계마다 DebugPassThrough 추가
summerized_rag_chain_debug = {
    "context": summerized_retriever,                    # 컨텍스트를 가져오는 retriever
    "question": DebugPassThrough()        # 사용자 질문이 그대로 전달되는지 확인하는 passthrough
}  | DebugPassThrough() | ContextToText()|   contextual_prompt | model

print("================= rag_chain 불러오기 완료  ===============")

while True:
    query = input("질문을 입력하세요! 종료를 원한다면 exit을 입력하세요.")
    if query == "exit":
        break
    print("question: " + query)
    
    response = summerized_rag_chain_debug.invoke(query)
    print("RAG response : " + response.content)