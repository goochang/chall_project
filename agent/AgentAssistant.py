from langchain_core.tools import Tool  # 외부 도구나 API를 통합하기 위한 기본 클래스
from youtube_agent import YouTubeAssistant
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish  # 에이전트의 행동과 종료를 나타내는 클래스
from langchain.agents import (
    Tool,                    # 에이전트가 사용할 도구를 정의하는 클래스
    AgentExecutor,          # 에이전트의 실행을 관리하는 클래스
    LLMSingleActionAgent,   # 한 번에 하나의 행동을 수행하는 에이전트
    AgentOutputParser       # 에이전트 출력을 파싱하는 기본 클래스
)
from langchain_core.runnables import RunnableSequence  # 여러 컴포넌트를 순차적으로 실행하기 위한 클래스
from langchain_openai import ChatOpenAI
from typing import List, Union, Any, Dict, Literal  # 다양한 타입 힌팅 클래스들
from pydantic import BaseModel, Field  # 데이터 검증과 직렬화를 위한 Pydantic 라이브러리
from dataclasses import dataclass  # 데이터 클래스 생성을 위한 데코레이터
from langchain_core.output_parsers import JsonOutputParser  # LLM의 출력을 JSON 형식으로 파싱하는 도구

import os
import re
from langchain.chains import LLMChain  # LLM과 프롬프트를 연결하는 체인 클래스
from langchain_community.llms import OpenAI  # OpenAI의 LLM을 사용하기 위한 클래스
from youtube_agent import get_youtube_tool
from RagChain import get_ai_news_tool
import streamlit as st


class AgentAction(BaseModel):
    """
    에이전트의 행동을 정의하는 Pydantic 모델
    Pydantic은 데이터 검증 및 관리를 위한 라이브러리입니다.
    """
    # Literal을 사용하여 action 필드가 가질 수 있는 값을 제한합니다
    action: Literal["youtube_search", "news_search", "not_supported"] = Field(
        description="에이전트가 수행할 행동의 타입을 지정합니다",
    )
    
    action_input: str = Field(
        description="사용자가 입력한 원본 질의 텍스트입니다",
        min_length=1,  # 최소 1글자 이상이어야 함
    )
    
    search_keyword: str = Field(
        description="""검색에 사용할 최적화된 키워드입니다.
        영상 제목 관련 키워드 혹은 AI 관련 키워드일 경우 핵심 검색어를 포함하고,
        not_supported 액션의 경우 빈 문자열('')을 사용합니다""",
        examples=["봉누도", "푸가토"]  # 예시 제공
    )

class CustomOutputParser(AgentOutputParser):
    """LLM 출력을 파싱하는 커스텀 파서 클래스
    
    LLM의 출력을 분석하여 다음 행동(AgentAction) 또는 
    최종 답변(AgentFinish)으로 변환하는 파서
    """
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """LLM 출력 텍스트를 파싱하는 메소드
        
        Args:
            text (str): LLM이 생성한 출력 텍스트
        
        Returns:
            Union[AgentAction, AgentFinish]: 다음 행동 또는 최종 답변
        """
        # 최종 답변이 포함된 경우 처리
        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text
            )
        
        # Action과 Input이 포함된 경우 처리
        match = re.search(r"Action: (.*?)[\n]*Action Input: (.*)", text, re.DOTALL)
        if not match:
            return AgentFinish(
                return_values={"output": text},
                log=text
            )
        
        # 매칭된 결과를 AgentAction으로 반환
        action = match.group(1).strip()
        action_input = match.group(2).strip()
        return AgentAction(tool=action, tool_input=action_input, log=text)

@dataclass
class AgentAssistantConfig:
    """
    Agent Assistant의 설정을 관리하는 데이터 클래스
    데이터클래스는 설정값을 깔끔하게 관리하고 타입 검사를 제공합니다.
    
    Attributes:
        openai_api_key (str): YouTube Data API 접근을 위한 인증 키
        llm_model (str): 사용할 언어 모델의 이름 (예: gpt-4)
        temperature (float): 언어 모델의 창의성 조절 파라미터 (0.0 = 결정적, 1.0 = 창의적)
        not_supported_message (str): AI 관련이 아닌 질문에 대한 기본 응답 메시지
    """
    openai_api_key: str
    llm_model: str 
    temperature: float = 0.0
    not_supported_message: str = "죄송합니다. AI 관련 질문에 대해서만 관련 영상을 제공할 수 있습니다."


class AgentAssistant:
    """
    Agent 사용 결과를 제공하는 통합 어시스턴트
    이 클래스는 사용자 질의를 처리하고 관련 정보를 검색하는 핵심 기능을 제공합니다.
    """
    @classmethod
    def from_env(cls) -> "AgentAssistant":
        """
        환경 변수에서 설정을 로드하여 인스턴스를 생성하는 클래스 메서드
        이 방식을 사용하면 설정을 코드와 분리하여 관리할 수 있습니다.
        """
        config = AgentAssistantConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),  # 환경 변수에서 API 키 로드
            llm_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),  # 기본 모델 지정
            temperature=float(os.getenv("TEMPERATURE", "0.0")),  # 문자열을 float로 변환
        )
        return cls(config)

    def __init__(self, config):
        """
        모든 필요한 컴포넌트와 설정을 초기화합니다.
        """
        self.config = config

        # LangChain의 ChatOpenAI 모델 초기화
        self.llm = ChatOpenAI(temperature=config.temperature, model=config.llm_model)
        
        # JSON 출력 파서 설정
        self.output_parser = JsonOutputParser(pydantic_object=AgentAction)

        self.tools = [
            get_youtube_tool(), get_ai_news_tool()
        ]

        # 프롬프트 템플릿 설정
        self.prompt = PromptTemplate(
            input_variables=["input"],  # 템플릿에서 사용할 변수들
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()},
            template="""당신은 정확하고 상세한 답변을 제공하는 도우미입니다.
            사용 가능한 도구:
            1. news_search: AI 관련 뉴스를 검색하는 도구입니다.
            2. youtube_search: YouTube에서 영상을 검색하는 도구입니다.

            다음 규칙을 따르세요:
            1. AI 관련 뉴스 검색이 필요한 경우 반드시 news_search 도구를 사용하세요.
            2. 영상 검색이 필요한 경우 youtube_search 도구를 사용하세요.
            3. 도구의 출력을 반드시 최종 답변에 통합하세요.
            4. youtube_search를 사용할경우 영상url 항목 제일 마지막에 출력한다.

            응답 형식:
            1. 도구를 사용할 경우:
            Action: [news_search 또는 youtube_search]
            Action Input: [도구에 전달할 입력]

            2. 최종 답변의 경우:
            Final Answer: [도구 결과를 포함한 명확하고 상세한 답변]

            분석할 질의: {input}

            {format_instructions}""")
        
        # 실행 체인 생성
        # 프롬프트 -> LLM -> 출력 파서로 이어지는 처리 파이프라인
        self.chain = RunnableSequence(
            first=self.prompt,
            middle=[self.llm],
            last=self.output_parser
        )

    def process_query(self, query: str) -> str:
        """
        사용자 질문을 처리하고 적절한 응답을 생성하는 메인 메서드
        
        Args:
            query: 사용자의 질문 문자열
            
        Returns:
            str: 검색 결과 또는 에러 메시지
        """
        try:
            # LangChain 체인 실행하여 질의 분석
            result = self.chain.invoke({"input": query})
     
            # 분석 결과에서 필요한 정보 추출
            action = result["action"]  # 수행할 액션
            action_input = result["action_input"]  # 원본 사용자 입력
            search_keyword = result["search_keyword"]  # LLM이 추출한 최적화된 검색어

            # 디버깅을 위한 정보 출력
            print(f"\n검색어 분석 결과:")
            print(f"- 원본 질문: {action_input}")
            print(f"- 추출된 키워드: {search_keyword}")
            print(f"- 선택된 도구: {action}\n")

            # AI 관련 질의가 아닌 경우 지원하지 않는다는 메시지 반환
            if action == "not_supported":
                return self.config.not_supported_message
            
            # 관련 질의에 맞는 tool 실행
            for tool in self.tools:
                if action == tool.name:
                    return tool.func(search_keyword)

            return f"지원하지 않습니다. : {action}"
      
        except Exception as e:
            return f"처리 중 오류 발생: {e}"

if __name__ == "__main__":

    assistant = AgentAssistant.from_env()

    # 사용자 질문
    user_queries = [
        "봉누도 관련 영상을 찾아줘",
        "'푸가토'라는 ai 모델 설명해줘",
    ]
    
    for query in user_queries:
        print(f"질문: {query}")
        result = assistant.process_query(query)
        print(f"답변: {result}\n")

