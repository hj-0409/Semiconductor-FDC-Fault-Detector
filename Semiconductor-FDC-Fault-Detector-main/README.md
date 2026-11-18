# 🚨 Semiconductor Fab Sensor FDC & AI Diagnosis Platform

> **Real-time Fault Detection & Classification (FDC) Dashboard for Semiconductor Manufacturing Processes powered by Machine Learning and Gemini AI.**

---

## 📌 1. 프로젝트 개요 (Overview)
본 프로젝트는 반도체 제조 공정 중 발생하는 고차원 센서 데이터(UCI SECOM)를 활용하여 챔버 내 이상 징후를 실시간으로 탐지하고 분류하는 **FDC(Fault Detection & Classification)웹 대시보드 플랫폼**입니다. 
단순한 통계 분석을 넘어, **PCA(주성분 분석)**를 통한 차원 축소와 **로지스틱 회귀(Logistic Regression)** 기반의 머신러닝 파이프라인, 그리고 **Google Gemini AI 설비 진단 리포트**를 결합하여 실무 현장에서 즉시 활용할 수 있도록 설계되었습니다.

---

## 🛠️ 2. 핵심 기술 스택 (Tech Stack)
* **Language**: Python 3.11+
* **Web Framework**: Streamlit (Cloud Deployment)
* **Machine Learning & Data Processing**: Scikit-learn (PCA, StandardScaler, Logistic Regression), Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn
* **AI Diagnosis Integration**: Google Gemini API (with Auto-Retry Error Handling for 503 Traffic spikes)

---

## ⚙️ 3. 주요 기능 (Key Features)
1. **실시간 센서 데이터 전처리 및 분석**
   * 590차원의 노이즈가 포함된 UCI SECOM 센서 데이터를 정제(결측치 중앙값 대체, 상수 센서 제거)하고 표준화(StandardScaler) 수행.
2. **인터랙티브 하이퍼파라미터 튜닝 (사이드바 제어판)**
   * **PCA 주성분 개수 조절**: 차원 압축에 따른 누적 분산율(Scree Plot) 변화를 실시간 확인.
   * **불량 판정 임계값(Threshold) 조절**: Recall(재현율)과 Precision(정밀도) 간의 트레이드오프를 직접 튜닝하며 Confusion Matrix 시각화 확인.
3. **커스텀 데이터 업로드 (CSV)**
   * 사용자 고유의 센서 데이터와 Pass/Fail 라벨이 포함된 CSV 파일을 업로드하여 커스텀 FDC 분석 수행 가능.
4. **Gemini AI 기반 챔버 설비 진단 리포트**
   * FDC 분석 결과(검증 웨이퍼 수, 이상 감지 건수, ROC-AUC 등)를 기반으로 Gemini AI 엔지니어가 챔버 이상 원인(플라즈마 밀도 불안정, 파티클 오염 등)과 즉각적인 조치 방안을 진단.

---

## 🚀 4. 로컬 실행 방법 (Quick Start)

```bash
# 1. 저장소 클론
git clone [https://github.com/hj-0409/Semiconductor-FDC-Fault-Detector.git](https://github.com/hj-0409/Semiconductor-FDC-Fault-Detector.git)
cd Semiconductor-FDC-Fault-Detector

# 2. 패키지 설치
pip install -r requirements.txt

# 3. Streamlit 앱 실행
streamlit run fdc_app.py
[Raw Sensor Data (590 Ch)] 
       │
       ▼
[Data Cleaning & Standardization] 
       │
       ▼
[PCA Dimensionality Reduction] ──> Scree Plot (Cumulative Variance)
       │
       ▼
[Logistic Regression Classifier] ──> Confusion Matrix & Metrics (ROC-AUC, Recall)
       │
       ▼
[Gemini AI Chamber Diagnostic Engine] ──> Expert Troubleshooting Report
Developed by 전효재 (Jeon Hyojae) | Suncheon National University, Department of Electronic Engineering