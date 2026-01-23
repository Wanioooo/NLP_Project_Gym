# ===============================
# PureGym Sentiment Analysis Dashboard
# ===============================

import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from sklearn.metrics import confusion_matrix

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="PureGym Sentiment Dashboard",
    page_icon="🏋️",
    layout="wide"
)

st.title("🏋️ PureGym Customer Sentiment Analysis Dashboard")
st.write("Analyze customer reviews using AI-powered sentiment and emotion detection.")

# -------------------------------
# LABEL & EMOJI MAPS
# -------------------------------
label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

emoji_map = {
    "joy": "😄",
    "anger": "😡",
    "sadness": "😢",
    "fear": "😨",
    "surprise": "😲",
    "disgust": "🤢",
    "neutral": "😐"
}

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_models():
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )

    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=5
    )

    return sentiment_model, emotion_model


sentiment_model, emotion_model = load_models()

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def batch_predict(pipeline_model, texts, batch_size=8):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        preds = pipeline_model(
            batch,
            truncation=True,
            max_length=256
        )
        results.extend(preds)
    return results

# -------------------------------
# SINGLE REVIEW ANALYSIS
# -------------------------------
st.subheader("✍️ Single Review Analysis")

user_review = st.text_area("Enter your review:")

user_rating = st.radio(
    "Give a rating:",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: "⭐" * x,
    horizontal=True
)

if st.button("Analyze Review"):
    if user_review.strip():

        sentiment_result = sentiment_model(user_review)[0]
        sentiment_label = label_map[sentiment_result["label"]]
        sentiment_score = sentiment_result["score"]

        emotion_results = emotion_model(user_review)[0]
        emotion_dict = {
            e["label"]: round(e["score"] * 100, 2) for e in emotion_results
        }

        rating_sentiment = rating_to_sentiment(user_rating)

        st.subheader("🧠 Sentiment Result")
        st.write(f"**Predicted Sentiment:** {sentiment_label}")
        st.write(f"**Confidence:** {sentiment_score:.2f}")
        st.write(f"**Rating-based Sentiment:** {rating_sentiment}")

        st.subheader("🎭 Emotion Detection")

        df_emotion = pd.DataFrame({
            "Emotion": [f"{emoji_map.get(k,'')} {k.capitalize()}" for k in emotion_dict],
            "Score (%)": list(emotion_dict.values())
        }).sort_values("Score (%)")

        fig = px.bar(
            df_emotion,
            x="Score (%)",
            y="Emotion",
            orientation="h",
            text="Score (%)",
            title="Emotion Confidence (%)"
        )

        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(xaxis_range=[0, 100])

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("⚠️ Sentiment vs Rating Check")

        if sentiment_label != rating_sentiment:
            st.warning("Mismatch detected between AI sentiment and user rating!")
        else:
            st.success("Sentiment matches rating.")

# ======================================================
# 📁 BATCH REVIEW ANALYSIS
# ======================================================
st.header("📁 Batch Review Analysis (CSV Upload)")

uploaded_file = st.file_uploader("Upload a CSV file containing reviews:", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded dataset:")
    st.dataframe(df.head())

    text_column = st.selectbox(
        "Select the column that contains review text:",
        df.columns
    )

    # 🔧 FIX 1: slider MUST be here (df already exists)
    max_rows = st.slider(
        "Limit number of reviews to analyze (for performance):",
        min_value=100,
        max_value=min(2000, len(df)),
        value=500,
        step=100
    )

    if st.button("Analyze Dataset"):
        with st.spinner("Analyzing reviews... Please wait."):

            # 🔧 FIX 2: LIMIT rows here
            texts = (
                df[text_column]
                .astype(str)
                .apply(clean_text)
                .tolist()[:max_rows]
            )

            sentiments = sentiment_pipeline(
                texts,
                batch_size=32,
                truncation=True
            )

            emotions = emotion_pipeline(
                texts,
                batch_size=32,
                truncation=True
            )

            df_result = df.head(max_rows).copy()

            df_result["Sentiment"] = [
                "Positive" if s["label"] == "POSITIVE" else "Negative"
                for s in sentiments
            ]
            df_result["Sentiment_Score"] = [s["score"] for s in sentiments]

            df_result["Emotion"] = [e["label"] for e in emotions]
            df_result["Emotion_Score"] = [e["score"] for e in emotions]

        st.success("Batch analysis completed!")

        st.dataframe(df_result.head())

        # -------------------------------
        # Batch Summary Statistics
        # -------------------------------
        st.subheader("📊 Batch Analysis Summary")

        total_reviews = len(df_result)
        sentiment_counts = df_result["Sentiment"].value_counts(normalize=True) * 100

        positive_pct = sentiment_counts.get("Positive", 0)
        negative_pct = sentiment_counts.get("Negative", 0)

        dominant_emotion = df_result["Emotion"].value_counts().idxmax()

        st.write(f"*Total reviews analyzed:* {total_reviews}")
        st.write(f"*Positive reviews:* {positive_pct:.2f}%")
        st.write(f"*Negative reviews:* {negative_pct:.2f}%")
        st.write(f"*Dominant emotion:* {dominant_emotion.capitalize()}")

        if positive_pct > negative_pct:
            st.info("📌 Overall sentiment trend: Mostly Positive")
        else:
            st.info("📌 Overall sentiment trend: Mostly Negative")

        csv = df_result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results as CSV",
            csv,
            "sentiment_emotion_results.csv",
            "text/csv"
        )

    # -------------------------------
    # CONFUSION MATRIX
    # -------------------------------
    st.subheader("⚠️ Rating vs AI Sentiment Confusion Matrix")

    cm = confusion_matrix(
        df["rating_sentiment"],
        df["sentiment"],
        labels=["negative", "neutral", "positive"]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Rating Negative", "Rating Neutral", "Rating Positive"],
        columns=["AI Negative", "AI Neutral", "AI Positive"]
    )

    st.dataframe(cm_df)

    mismatch_rate = (df["rating_sentiment"] != df["sentiment"]).mean() * 100
    st.warning(f"{mismatch_rate:.1f}% of reviews show sentiment–rating mismatch")

    st.subheader("✅ Final Annotated Dataset")
    st.dataframe(df)

else:
    st.info("⬆️ Upload a CSV file to begin analysis")
