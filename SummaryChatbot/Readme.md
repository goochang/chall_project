### AI News Rag Chatbot
> 작업 완료일 : 2024/11/29  
- 개요: AI 관련 뉴스를 크롤링하여 요약한 뒤, Rag Chain 방식을 활용해  
  데이터를 저장하고 이를 기반으로 챗봇을 구현.  
![image](https://github.com/user-attachments/assets/889e033f-b980-4094-829e-58199ae94060)

- 크롤링 URL : https://www.aitimes.com/news/articleList.html  

### 실행 방법
```
1. 필요한 패키지 설치
pip install -r requirements.txt

2. .env 설정
.env 파일 생성 후 
발급 받은 OPENAI_API_KEY, YOUTUBE_API_KEY 키 input

3. 로컬 구동
python times_news.py
python news_summerized.py
streamlit run chatbot.py
```

### 활용 기술
- BeautifulSoup
- FAISS
- retriever
- langchain

### 구조
![image](https://github.com/user-attachments/assets/afc68cdd-10fc-4ad7-abc6-1f3d493494f4)
