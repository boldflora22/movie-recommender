import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_model_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_tfidf(df):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=30000)
    matrix = vectorizer.fit_transform(df["plot"].map(clean_model_text))
    return vectorizer, matrix


def normalize(arr):
    arr = np.asarray(arr, dtype=float)
    mx = arr.max()
    return arr / mx if mx > 0 else arr


def recommend_with_weights(df, tfidf_matrix, vectorizer, config):

    n = len(df)
    total_score = np.zeros(n)
    reasons = [[] for _ in range(n)]

    if config.get("genre_weights"):
        for i, raw_genres in enumerate(df["genres_raw"].fillna("")):
            genres = str(raw_genres).split("|")
            weight_sum = sum([config["genre_weights"].get(g, 0) for g in genres])
            if weight_sum != 0:
                total_score[i] += 0.4 * (weight_sum / len(genres))
                liked = [g for g in genres if config["genre_weights"].get(g, 0) > 0.3]
                if liked:
                    reasons[i].append(f"类型契合({','.join(liked)})")

    if config.get("mood_text"):
        query_vec = vectorizer.transform([clean_model_text(config["mood_text"])])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).ravel()
        total_score += 0.3 * normalize(sim_scores)
        for i in np.where(sim_scores > 0)[0]:
            reasons[i].append("剧情氛围相符")

    years = df["year"].to_numpy(dtype=float)
    if config.get("year_mode") == "hard":
        lo, hi = config["year_range"]
        total_score[(years < lo) | (years > hi)] = -999
    elif config.get("year_mode") == "soft" and config.get("recent_boost", 0) > 0:
        recency = normalize(years - years.min())
        total_score += 0.15 * config["recent_boost"] * recency
        for i in np.where(recency > 0.8)[0]:
            reasons[i].append("近年新片")

    if config.get("popularity_weight", 0) != 0:
        votes = df["vote_count"].fillna(0).to_numpy()
        ratings = df["rating"].fillna(0).to_numpy() / 10.0

        pop_score = normalize(np.log1p(votes))

        if config["popularity_weight"] > 0:
            total_score += 0.15 * config["popularity_weight"] * ratings * pop_score
            for i in np.where((pop_score > 0.8) & (ratings > 0.7))[0]:
                reasons[i].append("大众高分")
        else:
            total_score += 0.15 * abs(config["popularity_weight"]) * ratings * (1 - pop_score)
            for i in np.where((pop_score < 0.3) & (ratings > 0.6))[0]:
                reasons[i].append("冷门宝藏")


    ranked_indices = np.argsort(total_score)[::-1]
    results = []

    for idx in ranked_indices:
        if total_score[idx] <= -900 or total_score[idx] <= 0:
            break

        row = df.iloc[idx]
        reason_str = "；".join(dict.fromkeys(reasons[idx])) if reasons[idx] else "综合推荐"

        results.append({
            "title": row["title"],
            "year": int(row["year"]),
            "genre": row["primary_genre"],
            "rating": row.get("rating", "N/A"),
            "votes": int(row.get("vote_count", 0)),
            "score": round(float(total_score[idx]), 3),
            "reason": reason_str,
            "plot": str(row["plot"])[:150] + "..."
        })

        if len(results) >= config.get("top_n", 10):
            break

    return pd.DataFrame(results)
