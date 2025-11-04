import io
from typing import Optional, Tuple, List

import requests
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score,
)
import matplotlib.pyplot as plt


st.set_page_config(page_title="Spam Detector (Chapter03)", page_icon="📧", layout="wide")


# ---------- Data Loading Utilities ----------
LOCAL_DATASET_PATH = (
    "Hands-On-Artificial-Intelligence-for-Cybersecurity/Chapter03/datasets/sms_spam_no_header.csv"
)


def _normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Drop any 'Unnamed' columns regardless of original dtype of column index
    keep_mask = ["unnamed" not in str(c).lower() for c in df.columns]
    if not all(keep_mask):
        df = df.loc[:, keep_mask]
    # Normalize column names to lowercase strings
    cols = [str(c).lower() for c in df.columns]
    df.columns = cols

    if set(["label", "text"]).issubset(df.columns):
        out = df[["label", "text"]].copy()
    elif set(["v1", "v2"]).issubset(df.columns):
        out = df[["v1", "v2"]].copy(); out.columns = ["label", "text"]
    elif df.shape[1] >= 2:
        out = df.iloc[:, :2].copy(); out.columns = ["label", "text"]
    else:
        raise ValueError("Unsupported dataset schema. Expect two columns: label,text.")

    out["label"] = out["label"].astype(str).str.strip().str.lower()
    out["text"] = out["text"].astype(str).fillna("").str.strip()
    out = out.loc[out["text"].str.len() > 0]
    out = out.drop_duplicates(subset=["label", "text"])  # dedupe
    return out


@st.cache_data(show_spinner=False)
def load_dataset_from_local(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    return _normalize_dataset(df)


@st.cache_data(show_spinner=False)
def load_dataset_from_bytes(data: bytes) -> pd.DataFrame:
    # Try CSV without header first, then auto-detect delimiter
    buf = io.BytesIO(data)
    try:
        df = pd.read_csv(buf, header=None)
        return _normalize_dataset(df)
    except Exception:
        buf.seek(0)
        df = pd.read_csv(buf, sep=None, engine="python")
        return _normalize_dataset(df)


@st.cache_data(show_spinner=False)
def load_dataset_from_url(url: str, timeout: float = 15.0) -> pd.DataFrame:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return load_dataset_from_bytes(resp.content)


def builtin_sample_dataset() -> pd.DataFrame:
    data = [
        ("ham", "Ok lar... Joking wif u oni..."),
        ("ham", "I'll call you later"),
        ("spam", "Free entry in 2 a wkly comp to win FA Cup final tkts! Text FA to 87121."),
        ("spam", "WINNER!! You have won a £1000 cash prize. Call 09061701461 now."),
        ("ham", "Are we meeting today?"),
        ("spam", "URGENT! Your Mobile No 1234 won 2,000,000. Claim now."),
    ]
    df = pd.DataFrame(data, columns=["label", "text"])
    return _normalize_dataset(df)


# ---------- Model Utilities ----------
def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
            ("clf", MultinomialNB(alpha=0.1)),
        ]
    )


@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    y = (df["label"].str.lower() == "spam").astype(int)
    X = df["text"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = getattr(pipe, "predict_proba", lambda x: None)(X_test)
    y_scores = y_proba[:, 1] if y_proba is not None else None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, target_names=["ham", "spam"], zero_division=0),
        "confusion": confusion_matrix(y_test, y_pred).tolist(),
        "y_test": y_test.to_numpy(),
        "y_pred": y_pred,
        "proba_available": y_scores is not None,
        "y_scores": y_scores,
        "X_test_texts": X_test.to_numpy(),
    }
    return pipe, metrics


def predict_one(pipe: Pipeline, text: str, threshold: float = 0.5) -> Tuple[str, float]:
    if hasattr(pipe, "predict_proba"):
        spam_score = float(pipe.predict_proba([text])[0][1])
        label = "spam" if spam_score >= float(threshold) else "ham"
        return label, spam_score
    else:
        pred = int(pipe.predict([text])[0])
        return ("spam" if pred == 1 else "ham"), np.nan


# ---------- UI ----------
st.title("📧 垃圾郵件（Spam）偵測服務")

# Data source selection
st.subheader("選擇資料來源")
source = st.radio(
    "資料來源：",
    options=[
        "使用預設本機路徑",
        "上傳 CSV 檔",
        "從 URL 載入",
        "使用內建小樣本",
    ],
    index=2,
    horizontal=False,
)

df = None
source_desc = ""
if source == "使用預設本機路徑":
    st.caption("預設本機路徑：" + LOCAL_DATASET_PATH)
    try:
        df = load_dataset_from_local(LOCAL_DATASET_PATH)
        source_desc = f"本機檔案：{LOCAL_DATASET_PATH}"
    except FileNotFoundError:
        st.warning("找不到本機資料檔案。您可以改用上傳、URL 或內建小樣本。")
    except Exception as e:
        st.error(f"載入本機資料失敗：{e}")
elif source == "上傳 CSV 檔":
    up = st.file_uploader("上傳 CSV/TSV 純文字檔（兩欄：label,text）", type=["csv", "tsv", "txt"])
    if up is not None:
        try:
            data = up.read()
            df = load_dataset_from_bytes(data)
            source_desc = f"使用者上傳：{getattr(up, 'name', 'uploaded_file')}"
        except Exception as e:
            st.error(f"解析上傳檔案失敗：{e}")
elif source == "從 URL 載入":
    default_url = (
        "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/master/Chapter03/datasets/sms_spam_no_header.csv"
    )
    url = st.text_input("輸入資料集 URL：", value=default_url)
    if st.button("從 URL 載入"):
        try:
            df = load_dataset_from_url(url)
            source_desc = f"遠端 URL：{url}"
        except Exception as e:
            st.error(f"從 URL 載入失敗：{e}")
else:  # 使用內建小樣本
    if st.button("載入內建小樣本"):
        try:
            df = builtin_sample_dataset()
            source_desc = "內建小樣本（示範用途）"
        except Exception as e:
            st.error(f"建立內建樣本失敗：{e}")

if df is not None:
    if source_desc:
        st.caption("資料來源：" + source_desc)
    # Data preview
    st.subheader("資料概覽")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(df.head(10))
    with col2:
        st.write("資料筆數：", len(df))
        st.write("欄位：", list(df.columns))
        st.write("標籤分佈（圖表）：")
        label_counts = df["label"].value_counts().reindex(["ham", "spam"]).fillna(0).astype(int)
        st.bar_chart(label_counts)

    with st.spinner("訓練模型中…"):
        pipe, metrics = train_model(df, test_size=0.2)

    # Top: evaluation metrics
    st.subheader("資料集評估結果")
    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    st.text("Classification Report:\n" + metrics["report"])

    st.write("混淆矩陣（實際=列，預測=欄）")
    cm = np.array(metrics["confusion"], dtype=int)
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["ham", "spam"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["ham", "spam"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    st.pyplot(fig, use_container_width=False)

    # Curves and scored test set (no interactive inputs)
    if metrics.get("proba_available") and metrics.get("y_scores") is not None:
        y_true = metrics["y_test"]
        y_scores = np.asarray(metrics["y_scores"]).astype(float)

        st.markdown("### 模型曲線（ROC / PR）")
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        pr_df = pd.DataFrame({"Recall": pr_rec, "Precision": pr_prec})

        st.caption(f"ROC AUC = {roc_auc:.3f} | PR AUC (AP) = {ap:.3f}")
        r1, r2 = st.columns(2)
        with r1:
            st.line_chart(roc_df.set_index("FPR"))
        with r2:
            st.line_chart(pr_df.set_index("Recall"))

        st.markdown("### 下載測試集評分結果（閾值=0.5）")
        test_texts = metrics.get("X_test_texts")
        if test_texts is not None:
            y_pred_thr = (y_scores >= 0.5).astype(int)
            out = pd.DataFrame({
                "text": test_texts,
                "true_label": np.where(y_true == 1, "spam", "ham"),
                "spam_score": y_scores,
                "pred_at_threshold": np.where(y_pred_thr == 1, "spam", "ham"),
            })
            st.download_button(
                "下載 scored_test.csv",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="scored_test.csv",
                mime="text/csv",
            )

    # Single input for prediction
    st.subheader("即時預測（輸入郵件內容）")
    # Quick-fill examples
    example_spam = (
        "Free entry in 2 a wkly comp to win FA Cup final tkts! Text FA to 87121."
    )
    example_ham = "Ok lar... Joking wif u oni..."
    c1, c2 = st.columns(2)
    with c1:
        if st.button("填入 Spam 範例"):
            st.session_state["input_text"] = example_spam
    with c2:
        if st.button("填入 Ham 範例"):
            st.session_state["input_text"] = example_ham

    text = st.text_area("輸入郵件內容：", height=120, key="input_text")
    if st.button("判斷是否為垃圾訊息") and text.strip():
        label, score = predict_one(pipe, text, threshold=0.5)
        if label == "spam":
            st.error(f"預測：Spam（信心 {score:.2%}）" if not np.isnan(score) else "預測：Spam")
        else:
            st.success(f"預測：Ham（非垃圾）{f'，信心 {score:.2%}' if not np.isnan(score) else ''}")
else:
    st.info("無法載入資料集。請確認本機路徑正確。")
