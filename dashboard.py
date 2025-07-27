import streamlit as st
import pandas as pd
import os
import time
import logging
from pymongo import MongoClient
from datetime import datetime
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "real_time_review_analyzer"
COLLECTION_NAME = "processed_reviews"
REFRESH_INTERVAL = 60

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger('RealTimeDashboard')

class RealTimeDashboard:
    def __init__(self):
        pass

    # Load MongoDB data
    def load_data(self):
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        data = list(collection.find())
        df = pd.DataFrame(data)
        
        # Ensure datetime format is parsed correctly
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df
    
    def format_absa(slef, absa_dict):
        if isinstance(absa_dict, dict):
            return ', '.join([f"{k}: {v}" for k, v in absa_dict.items()])
        return "N/A"
    
    def get_latest_reviews_by_item(self, df, selected_category, selected_item, top_n=10):
        """
        Filters and formats the latest reviews for a given product category and item.

        Parameters:
            df (pd.DataFrame): The full DataFrame containing review data.
            selected_category (str): The product category to filter.
            selected_item (str/int): The item_id to filter.
            top_n (int): Number of latest records to return (default is 10).

        Returns:
            pd.DataFrame: A formatted DataFrame of the latest reviews.
        """
        # Step 1: Filter data
        filtered_df = df[
            (df['product_category'] == selected_category) & 
            (df['item_id'] == selected_item)
        ].copy()

        # Step 2: Sort by datetime descending
        filtered_df.sort_values(by='datetime', ascending=False, inplace=True)

        # Step 3: Take latest N entries
        latest_n = filtered_df.head(top_n)

        # Step 4: Format datetime for display
        latest_n_display = latest_n.copy()
        latest_n_display['datetime'] = latest_n_display['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Step 5: Apply ABSA formatting
        latest_n_display['ABSA'] = latest_n_display['aspect_sentiment'].apply(self.format_absa)

        # Step 6: Rename columns for UI display
        latest_n_display.rename(columns={
            'datetime': 'DateTime',
            'review': 'Review',
            'sentiment': 'Sentiment',
            'emotion': 'Emotion'
        }, inplace=True)

        return filtered_df, latest_n_display


    def plot_sentiment_distribution(self, filtered_df):
        """
        Generates and displays a pie chart for sentiment distribution using Plotly.

        Parameters:
            filtered_df (pd.DataFrame): DataFrame containing a 'sentiment' column.
        """
        if "sentiment" not in filtered_df.columns or filtered_df.empty:
            st.warning("No sentiment data available to plot.")
            return

        # Step 1: Count sentiment occurrences
        sentiment_counts = filtered_df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        # Step 2: Identify the sentiment with the highest count
        max_idx = sentiment_counts['Count'].idxmax()
        pull_values = [0.1 if i == max_idx else 0 for i in range(len(sentiment_counts))]

        # Step 3: Plot pie chart
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

        # Step 4: Customize trace and layout
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

        return fig_sentiment


    def plot_emotion_distribution(self, filtered_df):
        """
        Generates and displays a bar chart for emotion distribution.

        Parameters:
            filtered_df (pd.DataFrame): DataFrame containing an 'emotion' column.
        """
        if "emotion" not in filtered_df.columns or filtered_df.empty:
            st.warning("No emotion data available to plot.")
            return

        # Step 1: Define desired order of emotions
        emotion_order = ['Happiness', 'Excitement', 'Gratitude', 'Anger', 'Disgust', 'Sadness', 'Neutral']

        # Step 2: Count and reorder emotions
        emotion_counts = (
            filtered_df['emotion'].value_counts()
            .reindex(emotion_order)
            .reset_index()
            .dropna()
        )
        emotion_counts.columns = ['Emotion', 'Count']

        # Step 3: Create bar chart
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

        # Step 4: Customize layout and traces
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

        return fig_emotion

    def plot_aspect_wordcloud(slef, filtered_df):
        """
        Generates and displays a word cloud for aspect frequency from ABSA data.

        Parameters:
            filtered_df (pd.DataFrame): DataFrame containing an 'aspect_sentiment' column 
                                        with dictionaries of aspect: sentiment.
        """

        # Step 1: Extract all aspects from the dict entries
        all_aspects = []
        for item in filtered_df.get('aspect_sentiment', []):
            if isinstance(item, dict):
                all_aspects.extend(item.keys())

        # Step 2: Create frequency dictionary
        aspect_freq = pd.Series(all_aspects).value_counts().to_dict()

        if not aspect_freq:
            st.warning("No aspect data found for word cloud.")
            return

        # Step 3: Generate WordCloud from frequencies
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(aspect_freq)

        # Step 4: Plot with matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        
        return fig


    # function for dashboard
    def dashboard(self):

        # --- Refresh Button ---
        if st.button("🔄 Refresh Data"):
            logger.info("Dashboard refreshed....")
            st.rerun()

        # Load Data
        logger.info("Loading the MongoDB.")
        df = self.load_data()

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

        logger.info(f"Selected Category: {selected_category}, Item: {selected_item}")

        logger.info(f"Display most recent 10 reviews with analysis.")
        filtered_df, latest_10_display = self.get_latest_reviews_by_item(df, selected_category, selected_item)
        # Show Total Reviews
        st.markdown(f"### Total Reviews: {filtered_df.shape[0]}")

        # 🔍 Latest 10 Reviews with Analysis
        st.subheader("Latest 10 Reviews with Analysis")

        # Display as Streamlit DataFrame
        with st.container():
            st.dataframe(latest_10_display[['DateTime', 'Review', 'Sentiment', 'Emotion', 'ABSA']], 
                        use_container_width=True, hide_index=True)

        st.markdown("---")
        
        col1, col2 = st.columns(2)

        with col1:
            logger.info(f"Sentiment Analysis distribution pie chart for {selected_category}: {selected_item}")
            fig_sentiment = self.plot_sentiment_distribution(filtered_df)    
            st.plotly_chart(fig_sentiment, use_container_width=True)

        with col2:
            logger.info(f"Emotion Recoognition distribution bar chart for {selected_category}: {selected_item}")
            fig_emotion = self.plot_emotion_distribution(filtered_df)
            st.plotly_chart(fig_emotion, use_container_width=True)

        logger.info(f"Aspect Wordcloud chart for {selected_category}: {selected_item}")
        st.subheader("🧩 Aspect Word Cloud")
        fig_aspect = self.plot_aspect_wordcloud(filtered_df)
        st.pyplot(fig_aspect)

        with st.empty():
            logger.info("Dashboard auto refreshed....")
            time.sleep(REFRESH_INTERVAL)
            st.rerun()
