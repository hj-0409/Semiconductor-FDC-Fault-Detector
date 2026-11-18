import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import requests
import os

# 페이지 설정
st.set_page_config(page_title="Semiconductor FDC & AI Diagnosis Platform", layout="wide")

st.title("🚨 Semiconductor Fab Sensor FDC & AI Diagnosis Platform")
st.markdown("실제 UCI SECOM 공정 센서 데이터(590차원)를 기반으로 한 **반도체 팹 FDC(이상 감지 및 분류) 및 Gemini AI 설비 진단 대시보드**입니다.")

st.markdown("---")

# FDC 실행 버튼 및 대시보드 메인 영역
if st.button("🚀 FDC 공정 이상 감지 모델 실행 및 AI 진단 시작", type="primary"):
    with st.spinner("⏳ 590차원 센서 데이터를 로드하고 머신러닝 모델을 학습하는 중입니다..."):
        try:
            # 1. 데이터 로드
            data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
            label_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"

            features = pd.read_csv(data_url, sep=r"\s+", header=None)
            labels = pd.read_csv(label_url, sep=r"\s+", header=None)
            y = labels[0].values

            # 2. 전처리
            missing_ratio = features.isnull().mean()
            features_cleaned = features.loc[:, missing_ratio <= 0.5]
            features_cleaned = features_cleaned.fillna(features_cleaned.median())
            features_cleaned = features_cleaned.loc[:, features_cleaned.std() > 0]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features_cleaned)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.25, random_state=42, stratify=y
            )

            # 3. PCA 및 모델 학습
            n_components = 30
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            explained_variance = np.sum(pca.explained_variance_ratio_) * 100

            clf = LogisticRegression(penalty='l2', C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
            clf.fit(X_train_pca, y_train)

            y_pred = clf.predict(X_test_pca)
            y_prob = clf.predict_proba(X_test_pca)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
            fail_detected_count = int(np.sum(y_pred == 1))
            total_test_count = len(y_pred)

            st.success(f"✅ 모델 학습 및 평가 완료! (ROC-AUC: **{roc_auc:.4f}**, 누적 분산: **{explained_variance:.1f}%**)")

            # 지표 메트릭 표시
            col1, col2, col3 = st.columns(3)
            col1.metric("검증 웨이퍼 총 수", f"{total_test_count}개")
            col2.metric("감지된 공정 이상(Fail)", f"{fail_detected_count}개", delta_color="inverse")
            col3.metric("모델 ROC-AUC", f"{roc_auc:.4f}")

            # 4. 시각화 차트 생성 및 웹 출력
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
            axes[0].plot(range(1, n_components + 1), cum_var, marker='o', color='#1f77b4', linewidth=2)
            axes[0].axhline(y=cum_var[-1], color='r', linestyle='--', alpha=0.7, label=f'Total Variance ({cum_var[-1]:.1f}%)')
            axes[0].set_title('PCA Scree Plot (Cumulative Explained Variance)', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Number of Principal Components')
            axes[0].set_ylabel('Cumulative Variance (%)')
            axes[0].grid(True, linestyle=':', alpha=0.6)
            axes[0].legend()

            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1],
                        xticklabels=['Pred Pass', 'Pred Fail'], yticklabels=['Actual Pass', 'Actual Fail'])
            axes[1].set_title(f'FDC Fault Detection Confusion Matrix (AUC: {roc_auc:.2f})', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Predicted Wafer State')
            axes[1].set_ylabel('Actual Wafer State')

            plt.tight_layout()
            st.pyplot(fig)

            # 5. Gemini AI 진단 연동
            st.markdown("---")
            st.subheader("🤖 Gemini AI 설비 엔지니어 진단 리포트")
            
            GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
            if not GEMINI_API_KEY:
                st.warning("⚠️ 환경 변수에 `GEMINI_API_KEY`가 설정되지 않아 AI 진단을 호출할 수 없습니다. 로컬 환경 변수나 Streamlit Secrets를 확인해주세요.")
            else:
                prompt = (
                    "너는 반도체 양산 팹(Fab)의 설비기술 및 공정 이상 감지(FDC) 전문 엔지니어다.\n"
                    f"UCI SECOM 센서 데이터 분석 결과, 총 {total_test_count}개의 검증 웨이퍼 중 {fail_detected_count}개의 웨이퍼에서 "
                    f"공정 이상(Fail)이 감지되었다. (모델 ROC-AUC 성능: {roc_auc:.4f})\n\n"
                    "590차원 센서 데이터에서 PCA 차원 축소 및 분류 모델을 통해 이 이상 징후가 포착되었을 때, "
                    "1) 챔버 내부에서 발생할 수 있는 주요 물리적/설비적 원인(예: 플라즈마 밀도 불안정, RF 반사 전력 튀김, 챔버 내벽 파티클 오염, 가스 밸브 헌팅 등)과 "
                    "2) 설비 엔지니어로서 즉시 조치해야 할 대응 방안을 3줄 이내로 전문가답게 진단해 줘."
                )
                
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={GEMINI_API_KEY}"
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                
                with st.spinner("🤖 Gemini AI 소자/설비 엔지니어가 챔버 이상 원인을 분석 중입니다..."):
                    try:
                        res = requests.post(url, json=payload, timeout=30)
                        data = res.json()
                        if "candidates" in data:
                            ai_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                            st.success(ai_text)
                        else:
                            st.error(f"AI 응답 오류: {data}")
                    except Exception as e:
                        st.error(f"AI 진단 요청 실패: {e}")

        except Exception as e:
            st.error(f"FDC 분석 실행 중 에러가 발생했습니다: {e}")