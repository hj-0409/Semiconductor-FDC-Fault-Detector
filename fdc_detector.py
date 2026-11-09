import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print("=" * 60)
print("[1/4] 반도체 제조 공정(SECOM) 챔버 센서 데이터 로드 중...")
print("=" * 60)

# UCI 머신러닝 저장소의 SECOM 실제 공정 데이터셋 자동 다운로드
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
label_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"

features = pd.read_csv(data_url, sep=r"\s+", header=None)
labels = pd.read_csv(label_url, sep=r"\s+", header=None)
y = labels[0].values

print(f"▶ 원본 센서 규격: {features.shape[0]}개 웨이퍼 측정치, {features.shape[1]}개 센서 채널")

print("\n" + "=" * 60)
print("[2/4] 결측치 보정 및 센서 노이즈 필터링 전처리 진행...")
print("=" * 60)

# 1) 결측치 비율 50% 초과 채널 제거
missing_ratio = features.isnull().mean()
features_cleaned = features.loc[:, missing_ratio <= 0.5]

# 2) 잔여 결측치 중앙값(Median) 대체
features_cleaned = features_cleaned.fillna(features_cleaned.median())

# 3) 분산이 0인(값이 변하지 않는 고장 센서) 컬럼 제거
features_cleaned = features_cleaned.loc[:, features_cleaned.std() > 0]
print(f"▶ 노이즈/상수 센서 필터링 완료: 590개 -> {features_cleaned.shape[1]}개 유효 센서 채널 정제")

# 4) 표준 정규화 (Z-score Scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_cleaned)

# 학습/검증 데이터 분할 (불균형 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

print("\n" + "=" * 60)
print("[3/4] 고차원 센서 PCA 차원 축소 및 규제 분류 모델 학습...")
print("=" * 60)

# 상위 30개 주성분 기저 투영
n_components = 30
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

explained_variance = np.sum(pca.explained_variance_ratio_) * 100
print(f"▶ 상위 {n_components}개 주성분의 누적 설명 분산: {explained_variance:.2f}% 확보")

# L2 Ridge 페널티 및 클래스 가중치 불균형 보정 로지스틱 회귀 학습
clf = LogisticRegression(penalty='l2', C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
clf.fit(X_train_pca, y_train)

# 이상 진단 평가
y_pred = clf.predict(X_test_pca)
y_prob = clf.predict_proba(X_test_pca)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

print(f"\n[모델 진단 성능 지표]")
print(f"▶ FDC 이상 감지 모델 ROC-AUC 점수: {roc_auc:.4f}")
print("-" * 60)
print(classification_report(y_test, y_pred, target_names=["Pass (정상)", "Fail (이상)"]))

print("=" * 60)
print("[4/4] 엔지니어링 결과 시각화 차트 생성 중...")
print("=" * 60)

# 시각화 대시보드 플롯
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. PCA Scree Plot
cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
axes[0].plot(range(1, n_components + 1), cum_var, marker='o', color='#1f77b4', linewidth=2)
axes[0].axhline(y=cum_var[-1], color='r', linestyle='--', alpha=0.7, label=f'Total Variance ({cum_var[-1]:.1f}%)')
axes[0].set_title('PCA Scree Plot (Cumulative Explained Variance)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Number of Principal Components', fontsize=11)
axes[0].set_ylabel('Cumulative Variance (%)', fontsize=11)
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend()

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1],
            xticklabels=['Pred Pass', 'Pred Fail'], yticklabels=['Actual Pass', 'Actual Fail'])
axes[1].set_title(f'FDC Fault Detection Confusion Matrix (AUC: {roc_auc:.2f})', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Predicted Wafer State', fontsize=11)
axes[1].set_ylabel('Actual Wafer State', fontsize=11)

plt.tight_layout()
plt.savefig("fdc_analysis_result.png", dpi=300)
print("▶ 시각화 대시보드 저장 완료: 'fdc_analysis_result.png'")
