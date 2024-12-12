### AI News Rag & Youtube Search Chatbot
> 작업 완료일 : 2024/11/29

개요: AI 관련 뉴스를 크롤링하고 요약한 데이터와 YouTube Search API를 통합하여  
사용자의 질문에 적합한 검색 방식을 선택적으로 활용하는 Agent 방식을 적용합니다.    

이 프로젝트는 사용자의 질문에 따라 적절한 검색 수단을 선택(뉴스 데이터 또는 YouTube 검색 결과)하여  
최적의 답변을 제공하는 시스템을 목표로 합니다.  
![image](https://github.com/user-attachments/assets/9f89cc92-797b-4769-9044-758a1c17b459)

- 크롤링 URL : https://www.aitimes.com/news/articleList.html  

### 실행 방법
```
1. 필요한 패키지 설치
pip install -r requirements.txt

2. .env 설정
.env 파일 생성 후 
발급 받은 OPENAI_API_KEY, YOUTUBE_API_KEY 키 input

3. 로컬 구동
streamlit run main.py
```

### 활용 기술
- BeautifulSoup
- FAISS
- retriever
- langchain
- pydantic
- Youtube Search API

### 구조
![image](https://github.com/user-attachments/assets/d2b46027-d123-420e-aba6-2430519b9500)
