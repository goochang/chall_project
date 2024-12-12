import streamlit as st
import os
from openai import OpenAI
from AgentAssistant import AgentAssistant
import time

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"
if "messages" not in st.session_state:  # 입력값에 대한 메시지
    st.session_state["messages"] = []
if "agent" not in st.session_state:  # 입력값에 대한 메시지
    st.session_state["agent"] = AgentAssistant.from_env()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

st.markdown(
    """
    <style> 
        .stMainBlockContainer {
            max-width: none;
            margin: 0 auto;
            width: 75vw;
            position: relative;
        }
        .video_info_wrap {
            display: flex;
            gap: 30px;
        }
        .video_info .video_info_title {
            font-weight: bold;
            font-size: 15px;
        }
        .video_title span {
            font-weight: bold;
            font-size: 20px;
        }
        .stChatMessageAvatarAssistant+.stChatMessageContent .stVerticalBlock {
            color:red;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def display_video(video_list):
    with st.container():
        # st.markdown(
        #     f"""

        # """,
        #     unsafe_allow_html=True,
        # )

        for i, video in enumerate(video_list, 1):
            print(i, video)
            st.markdown(
                f"""
                <div class="video_title">
                    <span>{video["title"]}</span>
                </div>
                <div class="video_info_wrap">
                    <div class="video_iframe">
                        <iframe width="640" height="360" src="{video["url"].replace("watch?v=", "embed/")}" 
                        title="{video["title"]}" frameborder="0" allow="accelerometer; autoplay; 
                        clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
                        referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
                    </div>
                    <div class="video_info">
                        <div>
                            <div class="video_info_title">
                                <span>채널</span>
                            </div>
                            <div>
                                <span>{video["channelTitle"]}</span>
                            </div>
                        </div>
                        <div>
                            <div class="video_info_title">
                                <span>게시일</span>
                            </div>
                            <div>
                                <span>{video["publishTime"]}</span>
                            </div>
                        </div>
                        <div>
                            <div class="video_info_title">
                                <span>URL</span>
                            </div>
                            <div>
                                <span>{video["url"]}</span>
                            </div>
                        </div>
                        <div>
                            <div class="video_info_title">
                                <span>조회수</span>
                            </div>
                            <div>
                                <span>👀 {video["viewCount"]}회</span>
                            </div>
                        </div>
                        <div>
                            <div class="video_info_title">
                                <span>좋아요 수</span>
                            </div>
                            <div>
                                <span>❤ {video["likeCount"]}개</span>
                            </div>
                        </div>
                        <div>
                            <div class="video_info_title">
                                <span>설명</span>
                            </div>
                            <div>
                                <span>{video["description"]}</span>
                            </div>
                        </div>
                    </div>
                    
                </div>
            """,
                unsafe_allow_html=True,
            )


st.title("Agent 채팅봇")

query = st.chat_input("메시지를 입력해주세요", key="fixed_chat_input")
if query:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if type(message["content"]) == list:
                display_video(message["content"])
            else:
                st.write(message["content"])

    data = {"role": "user", "content": query}
    st.session_state["messages"].append(data)

    with st.chat_message("user"):  # 사용자 채팅 표시
        st.write(query)

    # 어시스턴트 메시지 출력
    with st.chat_message("assistant"):
        response = st.session_state["agent"].process_query(query)
        if type(response) == list:
            display_video(response)
        else:
            st.write(response)

        st.session_state["messages"].append({"role": "assistant", "content": response})
