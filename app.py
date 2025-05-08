import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Doc2Vec
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка данных и моделей
@st.cache_resource
def load_resources():
    # Загрузка датасета
    games_df = pd.read_parquet("source/games.parquet")
    
    # Загрузка моделей
    doc2vec_model = Doc2Vec.load("models/doc2vec.model")
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # Загрузка признаков
    combined_features = np.load("models/combined_features.npy")
    
    return games_df, doc2vec_model, scaler, combined_features

games_df, doc2vec_model, scaler, combined_features = load_resources()

# Функция рекомендаций
def recommend(game_title, top_n=10):
    try:
        idx = games_df[games_df['title'].str.lower() == game_title.lower()].index[0]
        target_vector = combined_features[idx].reshape(1, -1)
        similarities = cosine_similarity(target_vector, combined_features)[0]
        similar_indices = similarities.argsort()[::-1][1:top_n+1]
        return games_df.iloc[similar_indices][['title', 'original_price', 'reviews_summary', 'developer']]
    except:
        return pd.DataFrame()

# Интерфейс Streamlit
st.title("🎮 Game Recommendation Engine")
st.write("Рекомендательная система для видеоигр")

# Поиск игры
game_title = st.selectbox(
    'Выберите или введите название игры:',
    games_df['title'].sort_values().unique())

if st.button('Найти рекомендации'):
    with st.spinner('Ищем похожие игры...'):
        recommendations = recommend(game_title)
        
    if not recommendations.empty:
        st.success(f"Рекомендации для игры '{game_title}':")
        for i, row in recommendations.iterrows():
            st.markdown(f"""
            **{row['title']}**  
            *Разработчик*: {row['developer']}  
            *Цена*: {row['original_price']}  
            *Отзывы*: {row['reviews_summary']}
            """)
            st.divider()
    else:
        st.error("Игра не найдена или недостаточно данных для рекомендаций")

st.caption("© 2024 Game Recommendations. Все права защищены.")