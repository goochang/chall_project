import os
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

folder_path = "SummaryChatbot/ai_news_json"
new_path = "SummaryChatbot/summerized_news_json"
# all_articles = pd.DataFrame()

model = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=os.environ['OPENAI_API_KEY'])

# 폴더 없을시 생성
if not os.path.exists(new_path):
    os.makedirs(new_path)

# 디렉터리 내 모든 .json 파일 읽기
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):  # .json 파일만 처리
        file_path = os.path.join(folder_path, filename)
        
        # 각 파일을 열어서 JSON 데이터 읽어오기
        with open(file_path, 'r', encoding='utf-8') as file:
            article_json = json.load(file)

            # article_df = pd.DataFrame([article_json])  # JSON을 DataFrame으로 변환
            content = article_json.get("content")
            
            system_message = SystemMessage(content="You are a summerizing AI. Please answer in Korean. Please reply only based inputs.")
            messages = [system_message]

            messages.append(HumanMessage(content=content))
            response = model.invoke(messages)
            
            reply = response.content
            print(reply)
            messages.append(AIMessage(content=reply))

            title = os.path.basename(filename)

            article_json["content"] = reply

            # JSON 파일 저장
            with open(new_path + "\\" + title, 'w', encoding='utf-8') as file:
                json.dump(article_json, file, ensure_ascii=False, indent=4)

            print(f"파일 저장 완료: {title}")