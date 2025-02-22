import json
import faiss
import numpy as np
import whisper
import yt_dlp
import pandas as pdㅌ
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient

# 1. 유튜브 영상 데이터 (JSON 파일로 저장) -> 썸네일 이미지는 프론트엔드에서 id로 생성
youtube_data = [
    {"id": 1, "title": "14년차 UX/UI 디자이너가 말하는 절대 피해야하는 탈락각 포트폴리오", "url": "https://youtu.be/uYjXlhrHr_4?si=yoquZuSIMKibBA5c", "video_id": "uYjXlhrHr_4"},
    {"id": 2, "title": "Machine Learning Basics", "url": "https://youtu.be/abc456", "video_id": "abc456"},
    {"id": 3, "title": "Introduction to MongoDB", "url": "https://youtu.be/mongo789", "video_id": "mongo789"},
    {"id": 3, "title": "Introduction to MongoDB", "url": "https://youtu.be/mongo789", "video_id": "mongo789"},
    {"id": 3, "title": "Introduction to MongoDB", "url": "https://youtu.be/mongo789", "video_id": "mongo789"},
    {"id": 3, "title": "Introduction to MongoDB", "url": "https://youtu.be/mongo789", "video_id": "mongo789"},
    {"id": 3, "title": "Introduction to MongoDB", "url": "https://youtu.be/mongo789", "video_id": "mongo789"},
    {"id": 3, "title": "Introduction to MongoDB", "url": "https://youtu.be/mongo789", "video_id": "mongo789"},
    {"id": 3, "title": "Introduction to MongoDB", "url": "https://youtu.be/mongo789", "video_id": "mongo789"},
    {"id": 3, "title": "Introduction to MongoDB", "url": "https://youtu.be/mongo789", "video_id": "mongo789"},
    {"id": 3, "title": "Introduction to MongoDB", "url": "https://youtu.be/mongo789", "video_id": "mongo789"}
]


with open("youtube_data.json", "w") as f:
    json.dump(youtube_data, f, indent=4)

# 2. 텍스트 임베딩 및 벡터 DB 구축
model = SentenceTransformer('all-MiniLM-L6-v2')
video_titles = [video["title"] for video in youtube_data]
embeddings = model.encode(video_titles, convert_to_numpy=True)

# FAISS 벡터 데이터베이스 생성 및 삽입
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# 3. 유사도 검색 함수 (사용자 입력에 대한 추천)
def search_videos(query, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, top_k)
    results = [youtube_data[i] for i in I[0]]
    return results

# 4. MongoDB 연동 (이력서 데이터 저장 및 검색)
client = MongoClient("mongodb://localhost:27017/")
db = client["resume_db"]
collection = db["resumes"]

resume_data = {"name": "Ari Kim", "skills": ["Python", "Machine Learning", "MongoDB"]}
collection.insert_one(resume_data)

def get_resume(name):
    return collection.find_one({"name": name})

# 5. 비디오 오디오 추출 및 텍스트 변환
def save_video_audio(url, filename):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': filename,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def extract_video_text(audio_file):
    model = whisper.load_model("large")
    result = model.transcribe(audio_file)
    return result["text"]

# 6. 이력서 기반 추천 시스템
class RecommendVideo:
    def __init__(self, token: str, video_database: pd.DataFrame, top_n=6):
        self.token = token 
        self.video_database = video_database
        self.top_n = top_n
        self.resume_text = self._extract_resume_text()
        self.resume_vector = model.encode([self.resume_text], convert_to_numpy=True)
    
    def _extract_resume_text(self):
        resume_data = get_resume(self.token)
        return " ".join(resume_data["skills"])
    
    def recommend_videos(self):
        video_vectors = np.array([model.encode([video["title"]], convert_to_numpy=True) for video in youtube_data])
        D, I = index.search(self.resume_vector, self.top_n)
        recommendations = [youtube_data[i] for i in I[0]]
        return recommendations

# 7. 실행 테스트
if __name__ == "__main__":
    user_query = "Python programming"
    recommended_videos = search_videos(user_query)
    print("Recommended Videos:", recommended_videos)
    
    resume = get_resume("Ari Kim")
    print("Resume Data:", resume)
    
    recommender = RecommendVideo("Ari Kim", pd.DataFrame(youtube_data))
    print("Recommended Videos based on Resume:", recommender.recommend_videos())
