# ================= 主界面 =================
st.title("🎬 智能电影选片系统")
st.markdown("拖动左侧滑条，定制你的专属片单！")

# 初始化 session_state，用来记住推荐结果
if "recommend_results" not in st.session_state:
    st.session_state.recommend_results = None

# 只有当点击侧边栏的“开始推荐”时，才去计算
if st.sidebar.button("🚀 开始推荐", type="primary"):
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
        # 将算出来的 results 存入 session_state
        st.session_state.recommend_results = recommend_with_weights(df, tfidf_matrix, vectorizer, config)

# 只要 session_state 里有结果，就展示出来（不管是因为点了推荐，还是点了盲盒）
if st.session_state.recommend_results is not None:
    results = st.session_state.recommend_results
    
    if results.empty:
        st.warning("没有找到符合你要求的电影，试试放宽一下条件吧！")
    else:
        st.success(f"为你找到了 {len(results)} 部电影！")
        
        # 👇 盲盒按钮（和下面的 for 循环平级，都在 else 里面）
        if st.button("🎲 不想选？帮我随便挑一部！"):
            random_movie = results.sample(1).iloc[0]
            st.balloons() # 撒花特效
            st.success(f"✨ 命中注定就是它：**{random_movie['title']}** ({random_movie['year']})")
            st.info(f"理由：{random_movie['reason']}")
            st.write(random_movie['plot'])
            
        st.divider() # 加一条分割线，让排版更好看
        
        # 👇 循环展示结果卡片（注意缩进，它和 if st.button 是平级的）
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
