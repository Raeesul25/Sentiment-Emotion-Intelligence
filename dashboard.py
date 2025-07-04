import streamlit as st
import pandas as pd
import os
import time
from pymongo import MongoClient
from datetime import datetime
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "real_time_review_analyzer"
COLLECTION_NAME = "processed_reviews"
REFRESH_INTERVAL = 10

# Load MongoDB data
def load_data():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    data = list(collection.find())
    df = pd.DataFrame(data)
    
    # Ensure datetime format is parsed correctly
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

# WordCloud Plotting
def plot_wordcloud(words_dict):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words_dict)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# function for dashboard
def dashboard():

    # --- Refresh Button ---
    if st.button("🔄 Refresh Data"):
        st.rerun()

    # Load Data
    df = load_data()

    # UI: Filters
    # st.title("AI-Driven Sentiment & Emotion Intelligence - Dashboard")
    col1, col2 = st.columns(2)

    with col1:
        product_categories = df['product_category'].unique()
        selected_category = st.selectbox("Select Product Category", product_categories)

    with col2:
        # Filter item IDs based on selected category
        item_ids = df[df['product_category'] == selected_category]['item_id'].unique()
        selected_item = st.selectbox("Select Item ID", item_ids)

    # Filter Data by Category and Item
    filtered_df = df[(df['product_category'] == selected_category) & (df['item_id'] == selected_item)].copy()
    filtered_df.sort_values(by='datetime', ascending=False, inplace=True)

    # Show Total Reviews
    st.markdown(f"### Total Reviews: {filtered_df.shape[0]}")

    # 🔍 Latest 10 Reviews with Analysis
    st.subheader("Latest 10 Reviews with Analysis")
    latest_10 = filtered_df.head(10)

    def format_absa(absa_dict):
        if isinstance(absa_dict, dict):
            return ', '.join([f"{k}: {v}" for k, v in absa_dict.items()])
        return "N/A"

    # Format datetime column for display
    latest_10_display = latest_10.copy()
    latest_10_display['datetime'] = latest_10_display['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    latest_10_display['ABSA'] = latest_10_display['aspect_sentiment'].apply(format_absa)

    # Optional: Rename columns for better readability (optional)
    latest_10_display.rename(columns={
        'datetime': 'DateTime',
        'review': 'Review',
        'sentiment': 'Sentiment',
        'emotion': 'Emotion'
    }, inplace=True)

    # Display as Streamlit DataFrame
    with st.container():
        st.dataframe(latest_10_display[['DateTime', 'Review', 'Sentiment', 'Emotion', 'ABSA']], 
                    use_container_width=True, hide_index=True)

    st.markdown("---")
    
    col1, col2 = st.columns(2)

    with col1:
        # 📊 Pie Chart: Sentiment Distribution
        sentiment_counts = filtered_df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        # Dynamically set pull values to explode only the highest sentiment
        max_idx = sentiment_counts['Count'].idxmax()
        pull_values = [0.1 if i == max_idx else 0 for i in range(len(sentiment_counts))]

        fig_sentiment = px.pie(
            sentiment_counts,
            values='Count',
            names='Sentiment',
            title='Sentiment Distribution',
            hole=0.3,
            labels={'Count': 'Review Count'},
            color='Sentiment',
            color_discrete_map={
                'Positive': '#2ECC71',
                'Negative': '#E74C3C',
                'Neutral': '#F1C40F'
            }
        )
        fig_sentiment.update_traces(
            textinfo='label+percent',
            textfont_size=16,
            hoverinfo='label+value+percent',
            pull=pull_values,
            marker=dict(line=dict(color='#000000', width=2))
        )
        fig_sentiment.update_layout(
            showlegend=False,
            title_font=dict(size=20, color='#333'),
            margin=dict(t=40, b=20, l=0, r=0),
            transition=dict(duration=500)
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

    with col2:
        # 📊 Bar Chart: Emotion Recognition Results
        emotion_order = ['Happiness', 'Excitement', 'Gratitude', 'Anger', 'Disgust', 'Sadness', 'Neutral']
        emotion_counts = (
            filtered_df['emotion'].value_counts()
            .reindex(emotion_order)
            .reset_index()
            .dropna()
        )
        emotion_counts.columns = ['Emotion', 'Count']
        
        fig_emotion = px.bar(
            emotion_counts,
            x='Emotion',
            y='Count',
            text='Count',
            title='Emotion Distribution',
            color='Emotion',
            category_orders={'Emotion': emotion_order},
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_emotion.update_traces(
            texttemplate='%{text}',
            textposition='outside',
            marker_line_width=1.5,
            marker_line_color='black',
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>',
            textfont_size=14
        )
        fig_emotion.update_layout(
            showlegend=False,
            xaxis_tickangle=-30,
            yaxis_title='Number of Reviews',
            xaxis_title='Emotion Type',
            bargap=0.25,
            title_font=dict(size=20),
            margin=dict(t=40, b=20, l=0, r=0),
            transition=dict(duration=500)
        )
        st.plotly_chart(fig_emotion, use_container_width=True)

    # ☁️ Aspect WordCloud
    all_aspects = []
    for item in filtered_df['aspect_sentiment']:
        if isinstance(item, dict):
            all_aspects.extend(item.keys())

    aspect_freq = pd.Series(all_aspects).value_counts().to_dict()
    if aspect_freq:
        st.subheader("🧩 Aspect Word Cloud")
        plot_wordcloud(aspect_freq)
    else:
        st.warning("No aspect data available for selected product.")

    # with st.empty():
    #     time.sleep(REFRESH_INTERVAL)
    #     st.rerun()
