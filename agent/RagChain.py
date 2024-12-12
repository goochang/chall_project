from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.tools import Tool  # 외부 도구나 API를 통합하기 위한 기본 클래스

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOpenAI

import pandas as pd
import json
import os
from dotenv import load_dotenv
load_dotenv()

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
    return all_articles

def get_retriever(texts:str):
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

class DebugPassThrough(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        output = super().invoke(*args, **kwargs)
        # print("Debug Output:", output)
        return output
    
# 문서 리스트를 텍스트로 변환하는 단계 추가
class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config=None, **kwargs):  # config 인수 추가
        # context의 각 문서를 문자열로 결합
        context_text = "\n".join([doc.page_content for doc in inputs["context"]])
        return {"context": context_text, "question": inputs["question"]}

class NewsAgent:
    def __init__(self):
        summerized_articles = get_article("summerized_news_json").loc[:, "content"]
        summerized_retriever = get_retriever('\n\n'.join(summerized_articles))

        # 모델 초기화
        model = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=os.environ['OPENAI_API_KEY'])

        contextual_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Answer the question using only the following context.
            """),
            ("user", "Context: {context}\\n\\nQuestion: {question}")
        ])
        
        # RAG 체인에서 각 단계마다 DebugPassThrough 추가
        rag_chain_debug = {
            "context": summerized_retriever,                    # 컨텍스트를 가져오는 retriever
            "question": DebugPassThrough()        # 사용자 질문이 그대로 전달되는지 확인하는 passthrough
        }  | DebugPassThrough() | ContextToText()|   contextual_prompt | model

        self.chain = rag_chain_debug

    def news_action(self, query):
        response = self.chain.invoke(query)
        return response.content

def get_ai_news_tool() -> str:
    """
    사용자 질문에 따라 최신 AI 뉴스를 검색합니다.
    
    Returns:
        str: 검색 결과 문자열
    """
    news_agent = NewsAgent()  # NewsAgent는 별도로 구현된 뉴스 검색 클래스
    return Tool(
        name="news_search",  # 도구의 식별자
        func=lambda query: news_agent.news_action(query),
        description="AI 기사 검색이 필요한 경우 news_search 도구를 사용하세요."
    )