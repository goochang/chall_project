# LangChain 관련: LLM과 프롬프트 처리를 위한 라이브러리
from langchain_core.prompts import (
    PromptTemplate,
)  # 프롬프트 템플릿을 생성하고 관리하기 위한 클래스
from langchain_openai import ChatOpenAI  # OpenAI의 GPT 모델을 사용하기 위한 인터페이스
from langchain_core.tools import Tool  # 외부 도구나 API를 통합하기 위한 기본 클래스
from langchain_core.runnables import (
    RunnableSequence,
)  # 여러 컴포넌트를 순차적으로 실행하기 위한 클래스
from langchain_core.output_parsers import (
    JsonOutputParser,
)  # LLM의 출력을 JSON 형식으로 파싱하는 도구

# 파이썬 타입 힌팅을 위한 임포트
# 타입 힌팅은 코드의 가독성을 높이고 IDE의 자동완성 기능을 개선합니다
from typing import List, Union, Any, Dict, Literal  # 다양한 타입 힌팅 클래스들

# 유틸리티 라이브러리들
from datetime import datetime  # 날짜와 시간 처리를 위한 클래스
from dataclasses import dataclass  # 데이터 클래스 생성을 위한 데코레이터
from pydantic import BaseModel, Field  # 데이터 검증과 직렬화를 위한 Pydantic 라이브러리
import requests  # HTTP 요청 처리를 위한 라이브러리
import os  # 운영체제 관련 기능과 환경 변수 접근을 위한 라이브러리
from dotenv import load_dotenv  # .env 파일에서 환경 변수를 로드하는 함수

# .env 파일에서 환경 변수 로드
load_dotenv()


@dataclass
class YouTubeAssistantConfig:
    """
    YouTube Assistant의 설정을 관리하는 데이터 클래스
    데이터클래스는 설정값을 깔끔하게 관리하고 타입 검사를 제공합니다.

    Attributes:
        youtube_api_key (str): YouTube Data API 접근을 위한 인증 키
        llm_model (str): 사용할 언어 모델의 이름 (예: gpt-4)
        temperature (float): 언어 모델의 창의성 조절 파라미터 (0.0 = 결정적, 1.0 = 창의적)
        not_supported_message (str): AI 관련이 아닌 질문에 대한 기본 응답 메시지
    """

    youtube_api_key: str
    llm_model: str
    temperature: float = 0.0
    not_supported_message: str = (
        "죄송합니다. AI 관련 질문에 대해서만 관련 영상을 제공할 수 있습니다."
    )


class YouTubeAssistant:
    """
    YouTube 검색 결과를 제공하는 통합 어시스턴트
    이 클래스는 사용자 질의를 처리하고 관련 YouTube 영상을 검색하는 핵심 기능을 제공합니다.
    """

    @classmethod
    def from_env(cls) -> "YouTubeAssistant":
        """
        환경 변수에서 설정을 로드하여 인스턴스를 생성하는 클래스 메서드
        이 방식을 사용하면 설정을 코드와 분리하여 관리할 수 있습니다.
        """
        config = YouTubeAssistantConfig(
            youtube_api_key=os.getenv(
                "YOUTUBE_API_KEY", ""
            ),  # 환경 변수에서 API 키 로드
            llm_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),  # 기본 모델 지정
            temperature=float(os.getenv("TEMPERATURE", "0.0")),  # 문자열을 float로 변환
        )
        return cls(config)

    def __init__(self, config: YouTubeAssistantConfig):
        """
        YouTube Assistant 초기화
        모든 필요한 컴포넌트와 설정을 초기화합니다.
        """
        self.search_url = "https://www.googleapis.com/youtube/v3/search"
        self.video_url = "https://www.googleapis.com/youtube/v3/videos"
        self.config = config

    def search_videos(self, query: str, max_results: int = 5) -> str:
        """
        YouTube API를 사용하여 비디오를 검색하고 정보를 수집하는 메서드

        Args:
            query: 검색할 키워드
            max_results: 검색할 최대 비디오 수 (기본값: 5)

        Returns:
            str: 포맷팅된 검색 결과 또는 에러 메시지
        """
        params = {
            "key": self.config.youtube_api_key,
            "q": query,
            "max_results": max_results,
            "type": "video",
            "videoType": "any",
            "order": "relevance",
            "part": "snippet",
            "regionCode": "KR",
            "relevanceLanguage": "ko",
        }

        try:
            # API 요청
            response = requests.get(self.search_url, params=params)
            response.raise_for_status()

            video_data = response.json()
            video_list = []
            for video in video_data["items"]:
                try:
                    video_id = video["id"]["videoId"]
                    video_stats = self._get_video_stats(video_id)
                    video_stats = video_stats["items"][0]

                    video_time = video["snippet"]["publishedAt"]
                    date_str = video_time.replace(
                        "Z", "+00:00"
                    )  # Z를 UTC 오프셋으로 변환
                    date_obj = datetime.fromisoformat(date_str)
                    date_obj = date_obj.strftime("%Y년 %m월 %d일")

                    video_info = {
                        "title": video["snippet"]["title"],
                        "channelTitle": video["snippet"]["channelTitle"],
                        "description": video["snippet"]["description"],
                        "publishTime": date_obj,
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "viewCount": format(
                            int(video_stats["statistics"]["viewCount"]), ","
                        ),
                        "likeCount": format(
                            int(video_stats["statistics"]["likeCount"]), ","
                        ),
                    }
                    video_list.append(video_info)
                except Exception as e:
                    print("Exception1", e)
        except Exception as e:
            print("Exception2", e)
        if not video_list:
            return "검색된 영상의 상세 정보를 가져오는데 실패했습니다."

        return video_list
        # return self._format_results(query, video_list)

    def _get_video_stats(self, video_id: str) -> dict:
        """
        특정 비디오의 통계 정보를 조회하는 내부 메서드

        Args:
            video_id: YouTube 비디오 ID

        Returns:
            dict: 비디오 통계 정보 (조회수, 좋아요 수 등)
        """
        params = {
            "key": self.config.youtube_api_key,
            "id": video_id,
            "part": "statistics",
        }
        try:
            # API 요청
            response = requests.get(self.video_url, params=params)
            response.raise_for_status()

            data = response.json()

        except Exception as e:
            data = {}
            print(e)

        return data

    def _format_results(self, query: str, videos: list) -> str:
        """
        검색 결과를 보기 좋게 포맷팅하는 내부 메서드

        Args:
            query: 원본 검색어
            videos: 비디오 정보 리스트

        Returns:
            str: 포맷팅된 검색 결과 문자열
        """
        result = ""

        for video in videos:
            date_str = video["publishTime"].replace(
                "Z", "+00:00"
            )  # Z를 UTC 오프셋으로 변환
            date_obj = datetime.fromisoformat(date_str)
            date_obj = date_obj.strftime("%Y-%m-%d")

            # print(video)
            result += f"{video['title']}\n"
            result += f"  채널 : {video['channelTitle']}\n"
            result += f"  설명 : {video['description']}\n"
            result += f"  업로드 날짜 : {date_obj}\n"
            result += f"  조회수 : {video['viewCount']}\n"
            result += f"  좋아요 수 {video['likeCount']}\n"
            result += f"  {video['url']}\n"

        return result


def get_youtube_tool() -> Tool:
    """
    YouTube 도구를 생성하여 반환합니다.
    """
    assistant = YouTubeAssistant.from_env()
    return Tool(
        name="youtube_search",
        func=lambda query: assistant.search_videos(query, 5),
        description="YouTube에서 영상을 검색합니다. 적합한 키워드를 제공해주세요.",
    )


if __name__ == "__main__":
    assistant = YouTubeAssistant.from_env()

    videos = assistant.search_videos("봉누도 오승철 영상 검색해줘")
    print(videos)
