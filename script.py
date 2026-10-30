import streamlit as st
import pandas as pd
from engine import build_tfidf, recommend_with_weights

# 页面配置
st.set_page_config(page_title="智能电影选片", page_icon="🎬", layout="wide")


# 缓存加载数据，避免每次滑动条都重新读取，提升速度
@st.cache_resource
def load_data():
    df = pd.read_csv("movies_enriched.csv", encoding="utf-8-sig")
    # 确保空值不报错
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0)
    vectorizer, matrix = build_tfidf(df)
    return df, vectorizer, matrix


df, vectorizer, tfidf_matrix = load_data()

# ================= 侧边栏：用户偏好设置 =================
st.sidebar.header("🎯 你的观影偏好")

# 1. 类型滑条
st.sidebar.subheader("1. 类型偏好")
st.sidebar.caption("正值表示想看，负值表示不想看")
all_genres = ["science fiction", "action", "comedy", "romance", "horror", "thriller", "crime", "drama"]
genre_weights = {}
for g in all_genres:
    # 只显示在数据里存在的类型
    if g in df["primary_genre"].values:
        genre_weights[g] = st.sidebar.slider(f"{g.capitalize()}", -1.0, 1.0, 0.0, 0.1)

# 2. 年份设置
st.sidebar.subheader("2. 上映年份")
year_mode = st.sidebar.radio("年份限制模式", ["不限", "优先近年新片", "严格限定年份"])
if year_mode == "严格限定年份":
    year_range = st.sidebar.slider("选择年份区间", 1900, 2024, (2015, 2024))
    recent_boost = 0.0
    y_mode = "hard"
elif year_mode == "优先近年新片":
    recent_boost = st.sidebar.slider("新片加分强度", 0.0, 1.0, 0.5, 0.1)
    year_range = (1900, 2024)
    y_mode = "soft"
else:
    recent_boost = 0.0
    year_range = (1900, 2024)
    y_mode = "off"

# 3. 大众 vs 小众
st.sidebar.subheader("3. 口碑倾向")
st.sidebar.caption("← 偏小众宝藏 | 偏大众口碑 →")
popularity_weight = st.sidebar.slider(" ", -1.0, 1.0, 0.0, 0.1)

# 4. 氛围搜索
st.sidebar.subheader("4. 氛围关键词")
st.sidebar.caption("输入英文效果更好，如: mind-bending, romantic, funny")
mood_text = st.sidebar.text_input("例如：space adventure")

# 推荐数量
top_n = st.sidebar.number_input("推荐数量", 5, 30, 10)

# ================= 主界面 =================
st.title("🎬 智能电影选片系统")
st.markdown("拖动左侧滑条，定制你的专属片单！")

if st.sidebar.button("🚀 开始推荐", type="primary"):
    # 打包配置
    config = {
        "genre_weights": genre_weights,
        "year_mode": y_mode,
        "year_range": year_range,
        "recent_boost": recent_boost,
        "popularity_weight": popularity_weight,
        "mood_text": mood_text,
        "top_n": top_n
    }

    with st.spinner("正在为你筛选电影..."):
        results = recommend_with_weights(df, tfidf_matrix, vectorizer, config)

    if results.empty:
        st.warning("没有找到符合你要求的电影，试试放宽一下条件吧！")
    else:
        st.success(f"为你找到了 {len(results)} 部电影！")
     
    if st.button("🎲 不想选？帮我随便挑一部！"):
            random_movie = results.sample(1).iloc[0]
            st.balloons() # 撒花特效
            st.success(f"✨ 命中注定就是它：**{random_movie['title']}** ({random_movie['year']})")
            st.info(f"理由：{random_movie['reason']}")
            st.write(random_movie['plot'])
        # 循环展示结果卡片
        for _, row in results.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"### {row['title']} ({row['year']})")
                    st.markdown(f"**类型**: {row['genre'].capitalize()} | **推荐理由**: `{row['reason']}`")
                    st.text(row['plot'])
                with col2:
                    st.metric("匹配度得分", f"{row['score']}")
                    # 处理没有评分的情况
                    rating_display = f"⭐ {row['rating']}/10" if row['rating'] > 0 else "暂无评分"
                    st.metric("IMDb 评分", rating_display, f"{row['votes']:,} 人评价")
