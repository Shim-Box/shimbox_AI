📂 폴더 구조
project/
├─ data/                       # 더미 데이터 / 학습용 CSV
├─ data_utils/
│   ├─ api_client.py           # API 연동
│   └─ dummy_generator.py      # 학습용 더미 데이터 생성
├─ ml/
│   ├─ train_patchtst.py        # PatchTST 학습
│   ├─ train_rf.py              # RandomForest 학습
│   ├─ inference.py             # 내일 물량 예측
│   └─ model_loader.py          # 모델 로드 도구
├─ models/
│   ├─ patchtst_cap/            # PatchTST 모델 저장
│   └─ rf_capacity.pkl          # RandomForest 모델 저장
├─ scripts/
│   ├─ init_products_from_excel.py   # 엑셀 → 상품 생성 API 호출
│   └─ assign_tomorrow.py            # 내일 상품 자동 배정 (메인)
├─ utils/
│   ├─ env.py                   # 환경 변수 로드
│   └─ logger.py                # 로깅 유틸
├─ .env
├─ requirements.txt
└─ README.md

-------------------------------------------------

⚙️ 환경 설정 (.env)
API_BASE_URL=http://your-api-server.com
ADMIN_EMAIL=admin@gmail.com
ADMIN_PASSWORD=12341234

-------------------------------------------------

📦 패키지 설치
pip install -r requirements.txt

-------------------------------------------------

🚀 실행 순서

1️⃣ 학습용 더미 생성
python -m data_utils.dummy_generator

생성 파일: data/train_history.csv

-------------------------------------------------

2️⃣ PatchTST 학습 (시계열 예측)
python -m ml.train_patchtst

생성
models/patchtst_cap/config.json
models/patchtst_cap/pytorch_model.bin

-------------------------------------------------

3️⃣ RandomForest 학습
python -m ml.train_rf

생성
models/rf_capacity.pkl
models/rf_feature_names.txt

-------------------------------------------------

4️⃣ 엑셀 상품 → DB에 저장
엑셀 경로: data/products.xlsx
python -m scripts.init_products_from_excel

-------------------------------------------------

5️⃣ 내일 물량 자동 배정 실행
python -m scripts.assign_tomorrow