# AI Study RAG

로컬 문서와 강의 자료를 색인하고 질문할 수 있는 AI 학습용 RAG 샌드박스입니다.

Qdrant 벡터DB, FastAPI 웹 UI, 문서 인덱서, OCR, 음성/영상 STT, Gemini 기반 답변 생성을 Docker Compose로 실행합니다.

## 주요 기능

- PDF, DOCX, PPTX, XLSX, Markdown, TXT 문서 색인
- HWP/HWPX 텍스트 추출
- PDF 내장 텍스트가 부족한 페이지 OCR 처리
- 오디오/영상 파일을 Whisper로 STT 처리
- dense embedding + sparse BM25 하이브리드 검색
- FlashRank 기반 검색 결과 재정렬
- Gemini API를 사용한 근거 기반 한국어 답변 생성
- 웹 UI에서 질문, 출처 확인, 재색인 실행

## 디렉터리 구조

```text
.
├── data/          # 개인 원본 자료. Git 업로드 제외
├── cache/         # 모델/파싱/STT 캐시. Git 업로드 제외
├── vectordb/      # Qdrant 저장소. Git 업로드 제외
├── workspace/     # 생성 결과/작업 파일. Git 업로드 제외
├── config/        # 로컬 실행 권한 설정
├── scripts/       # 인덱싱, 검색, 답변, 웹 서버 코드
└── docker-compose.yml
```

## 준비 사항

- Docker Desktop
- Gemini API key

환경 파일을 만듭니다.

```bash
cp .env.example .env
```

`.env`에 API 키를 설정합니다.

```text
GEMINI_API_KEY=your_api_key_here
```

`.env`, `data/`, `cache/`, `vectordb/`, `workspace/`는 개인 정보와 생성 데이터가 들어갈 수 있어 Git에 올리지 않습니다.

## 실행

```bash
docker compose up -d
```

웹 UI:

```text
http://localhost:8088
```

Qdrant dashboard:

```text
http://localhost:6333/dashboard
```

## 자료 추가

프로젝트별로 `data/` 아래에 폴더를 만듭니다.

```text
data/inbox
data/security-study
data/contracts
```

지원 파일 형식:

```text
pdf, docx, pptx, xlsx, md, txt, hwp, hwpx
mp3, m4a, wav, flac, aac, ogg, opus
mp4, mov, mkv, avi, webm, m4v
```

## 수동 색인

```bash
docker compose --profile tools run --rm rag \
  python /sandbox/scripts/index_pdfs.py --project inbox --skip-unchanged
```

강제로 새로 색인하려면:

```bash
docker compose --profile tools run --rm rag \
  python /sandbox/scripts/index_pdfs.py --project inbox --recreate
```

## 검색 테스트

```bash
docker compose --profile tools run --rm rag \
  python /sandbox/scripts/ask.py --project inbox --query "접근통제의 개념을 설명해줘"
```

## Gemini 답변 생성

```bash
docker compose --profile tools run --rm rag \
  python /sandbox/scripts/answer.py \
  --project inbox \
  --query "접근통제 절차는 무엇인가" \
  --show-context
```

## 웹 API

상태 확인:

```bash
curl 'http://localhost:8088/api/status?project=inbox'
```

질문:

```bash
curl -X POST 'http://localhost:8088/api/ask' \
  -H 'Content-Type: application/json' \
  -d '{"project":"inbox","query":"접근통제의 개념을 설명해줘","limit":6}'
```

## 보안 메모

이 저장소에는 개인 학습 자료와 API 키를 올리지 않습니다. 공개 저장소로 전환하기 전에는 `.gitignore`가 적용된 상태에서 `git status`로 스테이징 파일을 반드시 확인하세요.
