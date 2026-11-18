import os
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. 제미나이 API 키 연동
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def get_ai_fdc_diagnosis(fail_count, total_test_count, auc_score):
    """PCA 및 로지스틱 회귀로 감지된 팹 센서 이상(Fail)에 대해 AI 엔지니어 진단을 요청하는 함수 (자동 재시도 포함)"""
    if not GEMINI_API_KEY:
        return "⚠️ GEMINI_API_KEY가 설정되지 않아 AI 설비 진단을 실행할 수 없습니다."

    prompt = (
        "너는 반도체 양산 팹(Fab)의 설비기술 및 공정 이상 감지(FDC) 전문 엔지니어다.\n"
        f"UCI SECOM 센서 데이터 분석 결과, 총 {total_test_count}개의 검증 웨이퍼 중 {fail_count}개의 웨이퍼에서 "
        f"공정 이상(Fail)이 감지되었다. (모델 ROC-AUC 성능: {auc_score:.4f})\n\n"
        "590차원 센서 데이터에서 PCA 차원 축소 및 분류 모델을 통해 이 이상 징후가 포착되었을 때, "
        "1) 챔버 내부에서 발생할 수 있는 주요 물리적/설비적 원인(예: 플라즈마 밀도 불안정, RF 반사 전력 튀김, 챔버 내벽 파티클 오염, 가스 밸브 헌팅 등)과 "
        "2) 설비 엔지니어로서 즉시 조치해야 할 대응 방안을 3줄 이내로 전문가답게 진단해 줘."
    )
    
    # 최신 gemini-3.8-flash 모델명 적용
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    # 서버 과부하(503) 및 타임아웃 발생 시 최대 3번까지 재시도
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"\n🤖 AI 설비 엔지니어 분석 중... (시도 {attempt+1}/{max_retries})")
            res = requests.post(url, json=payload, timeout=30)
            data = res.json()
            
            if "candidates" in data:
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            
            if "error" in data and data["error"].get("code") == 503:
                print(f"⚠️ 구글 서버 트래픽 폭증 (503). 3초 뒤에 다시 시도합니다...")
                time.sleep(3)
                continue
            else:
                return f"구글 API 응답 에러 내용: {data}"
                
        except Exception as e:
            if attempt == max_retries - 1:
                return f"AI 진단 실패: {e}"
            print(f"⚠️ 네트워크 지연 발생. 3초 뒤에 다시 시도합니다...")
            time.sleep(3)
            
    return "AI 진단 실패: 서버 과부하로 인해 응답을 받지 못했습니다."

print("=" * 60)
print("[1/4] 반도체 제조 공정(SECOM) 챔버 센서 데이터 로드 중...")
print("=" * 60)

# UCI 머신러닝 저장소의 SECOM 실제 공정 데이터셋 로드
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
label_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"

features = pd.read_csv(data_url, sep=r"\s+", header=None)
labels = pd.read_csv(label_url, sep=r"\s+", header=None)
y = labels[0].values

print(f"▶ 원본 센서 규격: {features.shape[0]}개 웨이퍼 측정치, {features.shape[1]}개 센서 채널")

print("\n" + "=" * 60)
print("[2/4] 결측치 보정 및 센서 노이즈 필터링 전처리 진행...")
print("=" * 60)

missing_ratio = features.isnull().mean()
features_cleaned = features.loc[:, missing_ratio <= 0.5]
features_cleaned = features_cleaned.fillna(features_cleaned.median())
features_cleaned = features_cleaned.loc[:, features_cleaned.std() > 0]
print(f"▶ 노이즈/상수 센서 필터링 완료: 590개 -> {features_cleaned.shape[1]}개 유효 센서 채널 정제")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_cleaned)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

print("\n" + "=" * 60)
print("[3/4] 고차원 센서 PCA 차원 축소 및 규제 분류 모델 학습...")
print("=" * 60)

n_components = 30
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

explained_variance = np.sum(pca.explained_variance_ratio_) * 100
print(f"▶ 상위 {n_components}개 주성분의 누적 설명 분산: {explained_variance:.2f}% 확보")

clf = LogisticRegression(penalty='l2', C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
clf.fit(X_train_pca, y_train)

y_pred = clf.predict(X_test_pca)
y_prob = clf.predict_proba(X_test_pca)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

print(f"\n[모델 진단 성능 지표]")
print(f"▶ FDC 이상 감지 모델 ROC-AUC 점수: {roc_auc:.4f}")
print("-" * 60)
print(classification_report(y_test, y_pred, target_names=["Pass (정상)", "Fail (이상)"]))

# 이상 탐지 후 AI 연동
fail_detected_count = np.sum(y_pred == 1)
total_test_count = len(y_pred)

if fail_detected_count > 0:
    print(f"\n🚨 [FDC 경고] 검증 세트에서 공정 이상(Fail) 웨이퍼 {fail_detected_count}건 감지됨!")
    ai_report = get_ai_fdc_diagnosis(fail_detected_count, total_test_count, roc_auc)
    
    print("\n" + "=" * 60)
    print("📊 [Gemini AI 설비 엔지니어 진단 리포트]")
    print("=" * 60)
    print(ai_report)
    print("=" * 60)
else:
    print("\n✅ [판정] 검증 세트 내 이상 웨이퍼가 감지되지 않았습니다.")

print("=" * 60)
print("[4/4] 엔지니어링 결과 시각화 차트 생성 중...")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
axes[0].plot(range(1, n_components + 1), cum_var, marker='o', color='#1f77b4', linewidth=2)
axes[0].axhline(y=cum_var[-1], color='r', linestyle='--', alpha=0.7, label=f'Total Variance ({cum_var[-1]:.1f}%)')
axes[0].set_title('PCA Scree Plot (Cumulative Explained Variance)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Number of Principal Components', fontsize=11)
axes[0].set_ylabel('Cumulative Variance (%)', fontsize=11)
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend()

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1],
            xticklabels=['Pred Pass', 'Pred Fail'], yticklabels=['Actual Pass', 'Actual Fail'])
axes[1].set_title(f'FDC Fault Detection Confusion Matrix (AUC: {roc_auc:.2f})', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Predicted Wafer State', fontsize=11)
axes[1].set_ylabel('Actual Wafer State', fontsize=11)

plt.tight_layout()
plt.savefig("fdc_analysis_result.png", dpi=300)
print("▶ 시각화 대시보드 저장 완료: 'fdc_analysis_result.png'")
