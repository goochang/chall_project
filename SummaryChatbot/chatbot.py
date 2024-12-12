import os
from dotenv import load_dotenv
import streamlit as st
import RagChain
from langchain_openai import ChatOpenAI
from openai import OpenAI

# 환경 변수 로드
load_dotenv()
api_key = api_key=os.environ['OPENAI_API_KEY']
model = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=api_key)
client = OpenAI(api_key=api_key)

# Streamlit 기본 설정
st.header("AI 뉴스 챗봇")

# 채팅 히스토리 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 채팅 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 받기
if user_input := st.chat_input("메시지를 입력하세요"):
    # 사용자 메시지를 히스토리에 추가
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # RAG 모델 호출
    with st.chat_message("assistant"):
        try:
            
            # RAG 체인에서 각 단계마다 DebugPassThrough 추가
            summerized_rag_chain_debug = {
                "context": RagChain.summerized_retriever,                    # 컨텍스트를 가져오는 retriever
                "question": RagChain.DebugPassThrough()        # 사용자 질문이 그대로 전달되는지 확인하는 passthrough
            }  | RagChain.DebugPassThrough() | RagChain.ContextToText()|   RagChain.contextual_prompt | model

            chain_response = summerized_rag_chain_debug.invoke(user_input)

            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    *[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    *[
                        {"role": "system", "content": chain_response.content},
                        {"role": "user", "content": chain_response.content},
                    ],
                ],
                stream=True,
            )
            response = st.write_stream(stream)

            # 어시스턴트 메시지를 히스토리에 추가
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"RAG 호출 중 오류가 발생했습니다: {str(e)}")