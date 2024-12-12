### 1. AI News Rag Chatbot(Terminal)
> 작업 완료일 : 2024/11/29  
- 개요: AI 관련 뉴스를 크롤링하여 요약한 뒤, Rag Chain 방식을 
  활용해 데이터를 저장하고 이를 기반으로 챗봇을 구현.  
- 포함 내용 : url, title, created_at, content  
- LLM : openAI / chat-gpt-4o-mini  
- 활용 기술 : BeautifulSoup, FAISS, retriever, langchain  
- 크롤링 URL : https://www.aitimes.com/news/articleList.html  
