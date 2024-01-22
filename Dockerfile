# 베이스 이미지 선택
FROM python:3.8-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

# 포트 설정
EXPOSE 5000

# 애플리케이션 실행
CMD ["python", "./main.py"]