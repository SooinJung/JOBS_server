# 프로젝트 환경 설정을 중앙에서 관리하기 위한 파일
import os
from dotenv import load_dotenv

# 환경변수 정의
HOST = "0.0.0.0"
RHOST = HOST.replace(".", "\\.")
PORT = 8080
CPORT = 3000
ORIGIN_REGEX = f"^(https?://)?({RHOST}|localhost)(:({PORT}|{CPORT}|80|443))?(/.*)?$"

CURR_DIR = os.getcwd()
FILE_DIR = os.path.join(CURR_DIR, "files")
MAX_FSIZE = 50 * 1024 * 1024 # 50MB

# .env 파일 로드
load_dotenv()

# .env에서 가져온 api key
API_KEY = os.getenv("api_key")
PAFY_KEY = os.getenv("pafy_key")