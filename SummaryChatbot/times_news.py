#pip install requests
from typing import List, Dict
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
import requests

def get_ai_times(page):
    """
    AI 타임즈 기사를 가져오는 함수
    Args:
        page (int): 페이지
    Returns:
        list: 기사 리스트
    """

    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    headers =  {
        'User-Agent': user_agent
    }

    # 전체기사 URL - sm은 요약형
    base_url = "https://www.aitimes.com/news/articleList.html"
    
    # 검색어를 URL 인코딩하여 파라미터 구성
    params = {
        'page': page
    }
    
    results = []
    try:
        # API 요청
        response = requests.get(base_url, params=params, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        target_divs = soup.find_all("h4", class_ = "titles")
        
        for div in target_divs[:20]:
            get_url = div.find("a").get("href")
            results.append("https://www.aitimes.com/" + get_url)
                
        response.raise_for_status()
        return results
        
    except requests.exceptions.RequestException as e:
        print(f"에러 발생: {e}")
        return ""

def parse_info(url: str) -> json:
    """
    HTML에서 기사를 파싱하는 함수
    """
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    headers =  {
        'User-Agent': user_agent
    }

    try:
        # API 요청
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        heading_div = soup.find("header", class_="article-view-header")
        title = heading_div.find("h3", class_="heading").get_text(strip=True)
        created_at = heading_div.find_all("li")[4].get_text(strip=True)[3:]

        content_div = soup.find("article", id="article-view-content-div")
        articles = content_div.find_all("p") if content_div else []

        # 문자열만 추출하여 리스트 생성
        text_list = [p.get_text(strip=True) for p in articles]
        content = ' '.join(text_list)

        store = {
            "url" : url,
            "title": title,
            "created_at": created_at,
            "content": content
        }
        return store
        
    except requests.exceptions.RequestException as e:
        print(f"에러 발생: {e}")
        return ""

def main():
    print("<< AI 기사 가져오기 >>")
    url_results = []
    # AI 기사
    for page in range(1,2):
        url_list = get_ai_times(page)
        print(url_list)
        if url_list:
            url_results.extend(url_list)
            print(f"페이지 {page} 완료")
        
    if not url_results:
        print("AI 기사를 가져오는데 실패했습니다.")
        return
        
    print(f"\n총 {len(url_results)}개의 AI 기사를 수집했습니다.")
    # 데이터 저장

    article_path = "SummaryChatbot/ai_news_json"
    if not os.path.exists(article_path):
        os.makedirs(article_path)

    for url in url_results:
        article_json = parse_info(url)

        invalid_chars = r'\/:*?"<>|'
        title = article_json.get('title')

        # 파일 이름에서 부적절한 문자 제거
        for char in invalid_chars:
            title = title.replace(char, '_')
        file_name = f"ai_news_{title}.json"
        file_path = os.path.join(article_path, file_name)

        # JSON 파일 저장
        with open(article_path + "\\" + file_name, 'w', encoding='utf-8') as file:
            json.dump(article_json, file, ensure_ascii=False, indent=4)

        print(f"파일 저장 완료: {file_path}")
        # print( url)
    
if __name__ == "__main__":
    main()