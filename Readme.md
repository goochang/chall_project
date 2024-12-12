## 폴더 내부에 프로젝트별 Read.me 파일이 따로 있습니다.

### 1. AI News Rag Chatbot(Terminal)
> 작업 완료일 : 2024/11/29  
- 개요: AI 관련 뉴스를 크롤링하여 요약한 뒤, Rag Chain 방식을 
  활용해 데이터를 저장하고 이를 기반으로 챗봇을 구현.  
- 포함 내용 : url, title, created_at, content  
- LLM : openAI / chat-gpt-4o-mini  
- 활용 기술 : BeautifulSoup, FAISS, retriever, langchain  
- 크롤링 URL : https://www.aitimes.com/news/articleList.html  

### 2. AI News Rag & Youtube Search Chatbot(Streamlit)
> 작업 완료일 : 2024/11/29
- 개요: AI 관련 뉴스를 크롤링하고 요약한 뒤, Rag Chain 방식을 활용해 데이터를 저장하고 이를 기반으로 챗봇을 구현합니다.  
  또한, YouTube Search API와 통합하여 사용자의 질문에 적합한 검색 방식을 선택적으로 활용하는 Agent 방식을 적용합니다.  
  이 프로젝트는 사용자의 질문에 따라 적절한 검색 수단을 선택(뉴스 데이터 또는 YouTube 검색 결과)하여
  최적의 답변을 제공하는 시스템을 목표로 합니다.  
- LLM : openAI / chat-gpt-4o-mini
- 활용 기술 : BeautifulSoup, FAISS, retriever, langchain, pydantic, streamlit
- 크롤링 URL : https://www.aitimes.com/news/articleList.html
