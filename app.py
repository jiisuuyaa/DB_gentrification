import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import folium
from streamlit_folium import folium_static
import geopandas as gpd
from model import generate_risk_description  # 모델 관련 함수
from model2 import predict_risk, generate_ai_risk_description  # ✅ 모델 함수 불러오기


# 📂 CSV 파일 직접 로드
df = pd.read_csv("data/시군구.csv")

# 📌 Streamlit UI 설정
st.set_page_config(page_title="젠트리피케이션 위험 분석", layout="centered")

# 사이드바에서 페이지 선택
page = st.sidebar.radio("페이지 선택", ["서울시 행정동 젠트리피케이션 위험 분석", "다른 지역 젠트리피케이션 예측하기"])

if page == "서울시 행정동 젠트리피케이션 위험 분석":
    st.markdown(
    "<h1 style='text-align: center; font-size: 40px;'>📍 서울시 젠트리피케이션 위험 분석</h1>", 
    unsafe_allow_html=True
    )

    # 행정동 목록 가져오기
    dong_list = df["동"].unique().tolist()

    # 행정동 선택
    selected_dong = st.selectbox("🔍 분석할 지역을 선택하세요", dong_list)

    # 🔹 위험 그룹 (대분류) 매핑
    risk_group_map = {
        '일반': '🟢 일반',
        '주의': '🟡 주의',
        '경계': '🟠 경계',
        '위험_젠트리피케이션과열형': '🔴 위험_젠트리피케이션과열형',
        '위험_균형진행형': '🔴 위험_균형진행형',
        '위험_관심집중형': '🔴 위험_관심집중형',
        '위험_상권발달형': '🔴 위험_상권발달형'
    }

    # 선택한 지역의 위험 그룹 가져오기
    def get_risk_group(dong, df):
        row = df[df["동"] == dong]
        if not row.empty:
            cluster = row["클러스터"].values[0]
            return cluster, risk_group_map.get(cluster, '❓ 정보 없음')
        return None, "❓ 정보 없음"

    # 🗺️ GeoJSON 파일 로드
    geo_path = "data/HangJeongDong_ver20241001.geojson"
    seoul_map = gpd.read_file(geo_path)

    # 컬럼명 변경
    df.rename(columns={'행정동': 'adm_nm'}, inplace=True)

    # 데이터 병합
    merged = seoul_map.merge(df, on='adm_nm', how='left')

   # 분석 실행 버튼
    if st.button("분석 실행"):
        with st.spinner("🚀 분석 중... 잠시만 기다려 주세요!"):
            # ✅ 1. 위험 수준
            st.subheader("1️⃣ 위험 수준")
            cluster, risk_group = get_risk_group(selected_dong, df)
            st.write(f"📊 선택된 지역: **{selected_dong}**")
            st.write(f"🚨 {selected_dong} 위험 단계: **{risk_group}**")
            
            # ✅ 2. 위험 그룹 분포 시각화 
            st.subheader("2️⃣ 위험 그룹 분포")

            # 각 클러스터별 행정동 개수 세기
            cluster_counts = df["클러스터"].value_counts()

            # Matplotlib 그래프 생성
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['green'] * len(cluster_counts)

            # 선택한 행정동의 클러스터 강조 (색상 변경)
            if cluster in cluster_counts.index:
                selected_idx = cluster_counts.index.tolist().index(cluster)
                colors[selected_idx] = 'red'  # 선택된 행정동 클러스터를 빨간색으로 강조

            # 바 차트 생성
            bars = ax.bar(cluster_counts.index, cluster_counts.values, color=colors, alpha=0.8)

            # 그래프 설정
            ax.set_xlabel("위험 단계", fontsize=12)
            ax.set_ylabel("행정동 개수", fontsize=12)
            ax.set_title("서울시 위험 단계별 행정동 분포", fontsize=14)
            ax.set_xticklabels(cluster_counts.index, rotation=30, ha="right")

            # 데이터 라벨 추가
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom')

            # Streamlit에 그래프 표시
            st.pyplot(fig)

             # ✅ 3. 젠트리피케이션 지도
            st.subheader("3️⃣ 서울시 젠트리피케이션 위험 지도")

            # 지도 생성
            m = folium.Map(location=[37.5665, 126.9780], zoom_start=11, tiles="cartodb positron")

            # 젠트리피케이션 지수 범위 구하기
            gentrification_values = df["합산지수"].dropna()
            min_val, max_val = gentrification_values.min(), gentrification_values.max()

            thresholds = list(gentrification_values.quantile([0, 0.25, 0.5, 0.75, 1.0]))

            # Choropleth 적용 시 GeoDataFrame에서 직접 JSON 데이터 추출
            folium.Choropleth(
                geo_data=seoul_map.to_json(),  # ✅ GeoJSON을 문자열 형태로 변환
                data=df,
                columns=["adm_nm", "합산지수"],
                key_on="feature.properties.adm_nm",
                fill_color="YlOrRd",
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name="종합 젠트리피케이션 지수",
                threshold_scale=thresholds,
            ).add_to(m)

            # 선택된 행정동 강조하기
            if selected_dong:
                # 선택한 행정동의 GeoData 가져오기
                selected_area = merged[merged['동'] == selected_dong]

                if not selected_area.empty:
                    # 중심 좌표 구하기
                    centroid = selected_area.geometry.centroid.iloc[0]
                    m.location = [centroid.y, centroid.x]

                    # 강조된 행정동 경계 표시
                    folium.GeoJson(
                        selected_area.geometry,
                        style_function=lambda x: {
                            "fillColor": "cyan",
                            "color": "red",
                            "weight": 7,
                            "fillOpacity": 0.5,
                        },
                        name=f"{selected_dong} 강조"
                    ).add_to(m)

                    # 중심점에 마커 추가
                    folium.Marker(
                        location=[centroid.y, centroid.x],
                        popup=f"{selected_dong}",
                        icon=folium.Icon(color="red", icon="info-sign")
                    ).add_to(m)

            # 팝업 추가
            for _, row in merged.iterrows():
                popup_content = f"{row['adm_nm']}<br>합산 지수: {row['합산지수']:.2f}"
                folium.GeoJson(
                    row['geometry'],
                    popup=folium.Popup(popup_content, max_width=300),
                ).add_to(m)

            # Streamlit에 지도 표시
            folium_static(m)

            # ✅ 4. 선택된 지역 상세 분석
            st.subheader(f"4️⃣ {selected_dong} 상세 분석")
            result = generate_risk_description(selected_dong, df)
            st.write(result)

# ✅ [2] 다른 지역 젠트리피케이션 예측하기
elif page == "다른 지역 젠트리피케이션 예측하기":
    st.title("📊 다른 지역 젠트리피케이션 예측")

    # 사용자 입력 폼
    with st.form("predict_form"):
        st.write("AI를 이용해 원하는 지역의 현황 예측을 진행해주세요.")

        # 지역 선택 또는 입력란 추가
        selected_region = st.text_input("예측을 원하는 지역을 입력하세요:", value="")

        # 입력 순서를 학습 데이터와 동일하게 조정
        business_activation_index = st.number_input("상권 활성화 지수", min_value=-100.0, max_value=100.0, value=0.0)
        business_activation_sns_index = st.number_input("SNS 상권 활성화 지수", min_value=-100.0, max_value=100.0, value=0.0)
        gentrification_index = st.number_input("젠트리피케이션 지수", min_value=-100.0, max_value=100.0, value=0.0)
        gentrification_sns_index = st.number_input("SNS 젠트리피케이션 지수", min_value=-100.0, max_value=100.0, value=0.0)

        submitted = st.form_submit_button("예측하기")

    # ✅ 사용자가 폼 제출하면 모델 예측 실행
    if submitted:
        # 🔍 위험 단계 예측 실행
        risk_level, description = predict_risk(business_activation_index, business_activation_sns_index, gentrification_index, gentrification_sns_index)

        st.success(f"✅ 예측된 위험 단계: **{risk_level}**")
        st.write(description)

        # 🔹 AI 기반 추가 설명 생성
        with st.spinner("📡 AI가 입력한 정보를 바탕으로 분석 중 입니다."):
            ai_description = generate_ai_risk_description(risk_level, "사용자 입력 지역", gentrification_index, business_activation_index, gentrification_sns_index, business_activation_sns_index)
        
        st.write(f"💡 **{selected_region} 분석 결과:**")
        st.write(ai_description)