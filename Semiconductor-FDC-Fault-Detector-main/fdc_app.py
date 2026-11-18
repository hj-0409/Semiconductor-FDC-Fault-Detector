import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score
import requests
import os

# 페이지 설정
st.set_page_config(page_title="Semiconductor FDC & AI Diagnosis Platform", layout="wide")

st.title("🚨 Semiconductor Fab Sensor FDC & AI Diagnosis Platform")
st.markdown("반도체 팹 공정 센서 데이터를 기반으로 한 **실시간 FDC(이상 감지 및 분류) 및 Gemini AI 설비 진단 대시보드**입니다.")

# 기본 UCI SECOM 데이터 로드 (캐싱 적용)
@st.cache_data
def load_secom_data():
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
    label_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"
    features = pd.read_csv(data_url, sep=r"\s+", header=None)
    labels = pd.read_csv(label_url, sep=r"\s+", header=None)
    y = (labels[0].values == 1).astype(int)
    return features, y

# --- 사이드바 설정 ---
st.sidebar.header("⚙️ FDC 분석 설정")

uploaded_file = st.sidebar.file_uploader(
    "📂 커스텀 센서 데이터 업로드 (CSV)",
    type=["csv"],
    help="마지막 열이 불량 여부(0: 정상, 1: 불량)로 구성된 CSV 파일을 올리면 해당 데이터로 분석합니다. 비워두면 기본 UCI SECOM 데이터를 사용합니다."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ 모델 하이퍼파라미터 튜닝")
n_components = st.sidebar.slider("PCA 주성분 개수 (차원 압축)", min_value=5, max_value=100, value=30, step=5)
threshold = st.sidebar.slider(
    "불량 판정 임계값 (Threshold)",
    min_value=0.10, max_value=0.90, value=0.50, step=0.05,
    help="임계값을 낮추면 불량 감지율(Recall)이 올라가지만, 정상 웨이퍼 오탐지(False Alarm)도 증가합니다."
)

# --- 1. 데이터 로드 및 전처리 ---
try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        features = df.iloc[:, :-1]
        raw_y = df.iloc[:, -1].values
        y = (raw_y == 1).astype(int)
        st.info(f"📁 업로드된 파일(`{uploaded_file.name}`)을 기준으로 분석 중입니다. (총 {len(df)}개 웨이퍼, {features.shape[1]}개 센서)")
    else:
        with st.spinner("⏳ UCI SECOM 590차원 센서 데이터를 불러오는 중..."):
            features, y = load_secom_data()

    # 결측치 및 상수 센서 제거
    missing_ratio = features.isnull().mean()
    features_cleaned = features.loc[:, missing_ratio <= 0.5]
    features_cleaned = features_cleaned.fillna(features_cleaned.median())
    features_cleaned = features_cleaned.loc[:, features_cleaned.std() > 0]

    max_comp = min(features_cleaned.shape[0], features_cleaned.shape[1])
    actual_n_components = min(n_components, max_comp)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_cleaned)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    # --- 2. PCA 및 모델 학습 ---
    pca = PCA(n_components=actual_n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100

    clf = LogisticRegression(penalty='l2', C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
    clf.fit(X_train_pca, y_train)

    y_prob = clf.predict_proba(X_test_pca)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    fail_detected_count = int(np.sum(y_pred == 1))
    total_test_count = len(y_pred)

    # --- 3. 핵심 지표 표시 ---
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("검증 웨이퍼 수", f"{total_test_count}개")
    col2.metric("감지된 이상(Fail)", f"{fail_detected_count}개", delta=f"임계값 {threshold:.2f}", delta_color="off")
    col3.metric("불량 감지율 (Recall)", f"{recall*100:.1f}%")
    col4.metric("모델 ROC-AUC", f"{roc_auc:.4f}")

    # --- 4. 시각화 차트 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
    axes[0].plot(range(1, actual_n_components + 1), cum_var, marker='o', color='#1f77b4', linewidth=2)
    axes[0].axhline(y=cum_var[-1], color='r', linestyle='--', alpha=0.7, label=f'Total Variance ({cum_var[-1]:.1f}%)')
    axes[0].set_title(f'PCA Scree Plot (Top {actual_n_components} Components)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Number of Principal Components')
    axes[0].set_ylabel('Cumulative Variance (%)')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1],
                xticklabels=['Pred Pass', 'Pred Fail'], yticklabels=['Actual Pass', 'Actual Fail'])
    axes[1].set_title(f'Confusion Matrix (Threshold: {threshold:.2f}, AUC: {roc_auc:.2f})', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Wafer State')
    axes[1].set_ylabel('Actual Wafer State')

    plt.tight_layout()
    st.pyplot(fig)

    # --- 5. Gemini AI 설비 진단 ---
    st.markdown("---")
    st.subheader("🤖 Gemini AI 설비 엔지니어 진단 리포트")

    if st.button("🚀 현재 FDC 결과로 AI 설비 진단 리포트 생성", type="primary"):
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") if "GEMINI_API_KEY" in st.secrets else os.environ.get("GEMINI_API_KEY")

        if not GEMINI_API_KEY:
            st.warning("⚠️ `GEMINI_API_KEY`가 설정되지 않았습니다. Streamlit Cloud의 [Settings > Secrets]에 키를 등록해 주세요.")
        else:
            prompt = (
                "너는 반도체 양산 팹(Fab)의 설비기술 및 공정 이상 감지(FDC) 전문 엔지니어다.\n"
                f"센서 데이터 분석 결과(PCA {actual_n_components}차원 압축, 판정 임계값 {threshold:.2f}), "
                f"총 {total_test_count}개의 검증 웨이퍼 중 {fail_detected_count}개에서 공정 이상(Fail)이 감지되었다. "
                f"(Recall: {recall*100:.1f}%, Precision: {precision*100:.1f}%, ROC-AUC: {roc_auc:.4f})\n\n"
                "1) 챔버 내부에서 발생할 수 있는 주요 물리적/설비적 원인(플라즈마 밀도 불안정, RF 반사 전력 튀김, 파티클 오염, 가스 밸브 헌팅 등)과 "
                "2) 설비 엔지니어로서 즉시 조치해야 할 대응 방안을 3줄 이내로 전문가답게 진단해 줘."
            )

            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={GEMINI_API_KEY}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}

            with st.spinner("🤖 Gemini AI 설비 엔지니어가 챔버 이상 원인을 분석 중입니다 (트래픽 폭주 시 자동 재시도 중)..."):
                try:
                    import time
                    from requests.adapters import HTTPAdapter
                    from urllib3.util.retry import Retry

                    session = requests.Session()
                    retries = Retry(
                        total=3,
                        backoff_factor=1,
                        status_forcelist=[502, 503, 504],
                        raise_on_status=False
                    )
                    session.mount("https://", HTTPAdapter(max_retries=retries))

                    res = session.post(url, json=payload, timeout=30)
                    
                    if res.status_code == 200:
                        data = res.json()
                        if "candidates" in data:
                            ai_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                            st.success(ai_text)
                        else:
                            st.error(f"AI 응답 구조 오류: {data}")
                    else:
                        st.error(f"⚠️ 구글 AI 서버가 현재 혼잡합니다 (상태 코드: {res.status_code}). 잠시 후 다시 버튼을 눌러주세요.")
                except Exception as e:
                    st.error(f"AI 진단 요청 실패: {e}")

except Exception as e:
    st.error(f"FDC 분석 실행 중 에러가 발생했습니다: {e}")