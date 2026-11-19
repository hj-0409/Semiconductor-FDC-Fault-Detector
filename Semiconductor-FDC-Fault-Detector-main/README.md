# 🚨 Semiconductor Fab Sensor FDC & AI Diagnosis Platform

반도체 제조 공정 중 발생하는 590차원 센서 데이터(UCI SECOM)를 기반으로 챔버 이상 징후를 실시간 탐지·분류하고, **Gemini AI 설비 엔지니어**가 즉각적인 트러블슈팅 방안을 진단하는 Python 기반 지능형 FDC(Fault Detection & Classification) 웹 대시보드입니다.

🔗 **Live Demo (웹 서비스 바로가기)**: [https://semiconductor-fdc-fault-detector-gmr9qnat8z9qdb72sjcwsc.streamlit.app/](https://semiconductor-fdc-fault-detector-gmr9qnat8z9qdb72sjcwsc.streamlit.app/)

---

## 📌 1. 프로젝트 개요 (Overview)
실제 반도체 양산 팹(Fab)의 챔버에서는 수백 개의 센서 로그가 동시에 발생하며 결측치와 다중공선성으로 인해 공정 이상을 조기에 포착하기 어렵습니다. 
본 프로젝트는 **PCA(주성분 분석)** 차원 압축과 **로지스틱 회귀(Logistic Regression)** 파이프라인을 구축해 불량 웨이퍼를 판별하고, 엔지니어가 **판정 임계값(Threshold)**을 직접 튜닝하며 **Gemini AI 설비 진단 리포트**를 실시간으로 받아볼 수 있도록 설계되었습니다.

---

## 📊 2. 핵심 FDC 지표 및 설비 공학적 의미

| 제어/평가 지표 | 적용 방식 | 양산 팹(Fab) 현장 및 설비적 의미 |
| :--- | :--- | :--- |
| **PCA 주성분 개수** | 고차원 센서 데이터(590 Ch) 차원 압축 | 다중공선성 및 노이즈를 제거하고 핵심 분산 정보만 추출 |
| **판정 임계값 (Threshold)** | 사이드바 실시간 슬라이더 제어 ($0.10 \sim 0.90$) | 불량 유출 방지(Recall)와 설비 오탐지(False Alarm) 간 균형점 튜닝 |
| **불량 감지율 (Recall)** | 실제 불량 웨이퍼 중 모델이 잡아낸 비율 | 수율 방어를 위해 팹 현장에서 최우선으로 관리하는 핵심 지표 |
| **ROC-AUC** | 임계값 변화에 따른 전반적 분류 성능 | 클래스 불균형(정상 다수, 불량 극소수) 환경에서의 모델 변별력 검증 |

---

## ⚡ 3. 주요 기능 및 기술 스택
* **Language & Web:** Python 3.11+, Streamlit (Cloud Deployment)
* **ML & Data Pipeline:** Scikit-learn (`StandardScaler`, `PCA`, `LogisticRegression`), Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn (PCA Scree Plot & Confusion Matrix 실시간 연동)
* **AI Diagnosis & Stability:** Google Gemini API + `urllib3` 자동 재시도(Auto-Retry) 예외 처리로 503 트래픽 과부하 방어

1. **실시간 센서 데이터 전처리:** 결측률 50% 초과 센서 및 분산 0인 상수 센서 자동 필터링, 중앙값 대체 및 표준화 수행.
2. **인터랙티브 하이퍼파라미터 튜닝:** PCA 차원 수와 불량 판정 임계값(Threshold) 조절에 따른 성능 변화를 실시간 차트로 확인.
3. **커스텀 CSV 업로드:** 사용자 고유의 공정 센서 CSV 데이터를 업로드해 즉시 FDC 모델 재학습 및 검증 가능.
4. **Gemini AI 챔버 진단 리포트:** 감지된 불량 건수와 성능 지표를 동적 프롬프트에 바인딩해 물리적 이상 원인(플라즈마 불안정, RF 반사 전력, 파티클 등) 및 조치 방안 도출.

---

## 🖼️ 4. 분석 결과 시각화 (Preview)

![FDC Analysis Result](./fdc_analysis_result.png)

---

## 💡 5. 시스템 아키텍처 및 파이프라인 (System Architecture)

```text
[Raw Sensor Data (590 Ch)] 
       │
       ▼
[Data Cleaning & Standardization] ──> Missing > 50% & Zero-Variance Removal
       │
       ▼
[PCA Dimensionality Reduction] ──> Scree Plot (Cumulative Variance)
       │
       ▼
[Logistic Regression Classifier] ──> Threshold Tuning & Confusion Matrix (Recall, AUC)
       │
       ▼
[Gemini AI Chamber Diagnostic Engine (Auto-Retry)] ──> Expert Troubleshooting Report
```

---

## 🚀 6. 로컬 실행 방법 (Quick Start)

```bash
# 1. 저장소 클론
git clone https://github.com/hj-0409/Semiconductor-FDC-Fault-Detector.git
cd Semiconductor-FDC-Fault-Detector

# 2. 패키지 설치
pip install -r requirements.txt

# 3. Streamlit 앱 실행
streamlit run fdc_app.py
```

---
* Developed by **전효재 (Jeon Hyojae)** | Suncheon National University, Department of Electronic Engineering