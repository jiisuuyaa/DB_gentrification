import openai
import pandas as pd
import os
from dotenv import load_dotenv
import re
import streamlit as st

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 키 불러오기
openai.api_key = st.secrets["openai"]["api_key"]

# # OpenAI 클라이언트 설정
# client = openai.api_key


# ✅ 위험 단계별 설명 (사전 정의)
risk_explanations = {
    "위험_상권발달형": "🔴 이 지역은 상권이 발전했지만 SNS 활동이 적고 젠트리피케이션이 상대적으로 덜 진행된 지역입니다. 대표적으로 노량진, 대치동 등이 있습니다.",
    "위험_관심집중형": "🔴 이 지역은 SNS에서 큰 관심을 받고 있으며, 젊은 층이 많이 방문해 젠트리피케이션이 빠르게 진행 중입니다. 대표적으로 명동, 연남동 등이 있습니다",
    "위험_균형진행형": "🔴 이 지역은 상권, SNS 활동, 젠트리피케이션이 균형을 이루는 지역으로 점진적 변화가 예상됩니다. 대표적으로 상암동, 독산동 등이 있습니다.",
    "위험_젠트리피케이션과열형": "🔴 이 지역은 젠트리피케이션이 과열되었으나, 상권 활성화는 상대적으로 낮은 지역입니다. 대표적으로 신도림, 아현동 등이 있습니다.",
    "경계": "🟠 이 지역은 상권이 빠르게 성장 중이며, 젠트리피케이션 진행 가능성이 높은 지역입니다.",
    "주의": "🟡 이 지역은 상권이 서서히 성장하는 지역이며, 일부 지역에서 젠트리피케이션 변화가 시작될 가능성이 있습니다.",
    "일반": "🟢 이 지역은 상권이 안정적이며 젠트리피케이션 영향이 적은 지역으로, 임대료 상승 등의 변화도 미미한 편입니다."
}

# 📌 행정동 데이터 검색 함수
def retrieve_data(dong, df):
    dong_data = df[df['동'] == dong]
    if dong_data.empty:
        return None
    return dong_data.iloc[0].to_dict()

# 📏 사실성 점검 함수
def evaluate_factual_accuracy(generated_text, actual_data):
    score = 100
    for key, value in actual_data.items():
        if isinstance(value, (int, float)):
            pattern = re.search(rf"{key}\s*:\s*([\d\.]+)", generated_text)
            if pattern:
                generated_value = float(pattern.group(1))
                if value > 0:
                    error_rate = abs(generated_value - value) / value * 100
                    score -= min(error_rate, 100)
    return max(score, 0)

# 🔍 일관성 점검 함수
def evaluate_consistency(description, cluster):
    prompt = f"""
    다음 설명이 '{cluster}' 위험 단계와 논리적으로 일치하는지 100점 만점으로 평가해 주세요.
    - 위험 단계: {cluster}
    - 설명: {description}
    결과는 다음 형식으로 제공하세요:
    일관성 점수: [숫자]
    """
    try:
        # ✅ 최신 API 호출 방식
        client = openai.Client()

        response = client.chat.completions.create(
            model="gpt-4",  # 최신 모델로 변경
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        match = re.search(r"일관성 점수:\s*(\d+)", result)

        return int(match.group(1)) if match else 0

    except openai.error.InvalidRequestError as e:
        st.error(f"API 호출 실패: {str(e)}")
        return 0
        
# 📌 OpenAI GPT-4 기반 위험 설명 생성 함수
def generate_risk_description(dong_name, df, max_attempts=3, min_score=80):
    dong_info = retrieve_data(dong_name, df)
    if not dong_info:
        return f"⚠️ '{dong_name}'에 대한 데이터가 없습니다. 올바른 행정동을 입력해주세요."

    cluster = dong_info.get("클러스터", "알 수 없음")
    explanation = risk_explanations.get(cluster, "해당 위험 단계에 대한 정보가 없습니다.")

    trade_index = round(dong_info['상권활성화지수'], 2)
    gentrification_index = round(dong_info['젠트리피케이션지수'], 2)
    sns_trade = round(dong_info['상권활성화_SNS'], 2)
    sns_gentrification = round(dong_info['젠트리피케이션_SNS'], 2)

    # 🔹 "위험_"이 붙어 있는 경우 전체적으로 진행도가 높은 지역임을 강조
    if "위험_" in cluster:
        risk_comment = "📍 이 지역은 이미 다른 지역에 비해 상권 활성화와 젠트리피케이션이 상당히 진행된 상태입니다."
    else:
        risk_comment = "📍 이 지역은 아직 다른 지역에 비해 젠트리피케이션과 상권 활성화가 상대적으로 덜 진행된 곳입니다."

    # 🔹 같은 위험 단계를 가진 다른 서울 지역 찾기
    similar_districts = df[df["클러스터"] == cluster]["동"].unique().tolist()
    similar_districts = [d for d in similar_districts if d != dong_name]  # 자기 자신 제외

    if similar_districts:
        similar_places_text = f"🔹 **'{dong_name}'과(와) 같은 위험 단계를 가진 서울 지역:** {', '.join(similar_districts[:5])} 등"
    else:
        similar_places_text = "🔹 이 지역과 같은 위험 단계를 가진 다른 지역 정보가 부족합니다."


    # 🔹 OpenAI GPT-4 프롬프트 생성
    for attempt in range(max_attempts):
        prompt = f"""
        🏙️ **{dong_name}의 위험 분석**  

        🔎 이 지역은 **{cluster}** 단계에 해당합니다.  
        {risk_comment}  
        {explanation}  

        📢 **💡 젠트리피케이션과 상권 활성화란?**  
        젠트리피케이션(Gentrification)은 특정 지역의 경제가 성장하면서 건물 임대료가 오르고 기존 거주민들이 떠나게 되는 현상을 말합니다.  
        상권 활성화는 소비와 유동 인구가 증가하면서 지역 경제가 활발해지는 것을 뜻합니다.  

        📊 **📌 {dong_name}의 주요 경제 지표**  
        - 🏬 **상권 활성화 지수**: {trade_index}  
        - 🏙️ **젠트리피케이션 지수**: {gentrification_index}  
        - 📢 **SNS 상권 활성화 점수**: {sns_trade}  
        - 💬 **SNS 젠트리피케이션 점수**: {sns_gentrification}  

        🔎 **이제 위의 데이터를 바탕으로 {dong_name}의 경제적 위험성과 향후 전망을 상세히 분석해주세요요.**  
        - 만약 "{cluster}"에 "위험_"이 포함되어 있으면, 해당 지역은 다른 지역에 비해 상권 활성화와 젠트리피케이션이 상당히 진행된 상태입니다. 이를 고려하여 분석해주세요.
        - 현재 이 지역의 경제적 변화 수준이 다른 지역과 비교했을 때 어느 정도인지 설명해주세요.
        - 상권 활성화가 계속 진행될 가능성이 높은지, 젠트리피케이션이 앞으로 어떤 영향을 미칠지 분석해주세요요.
        - 지역 상권과 부동산 시장에 미치는 영향을 포함해서 전망을 이야기해주세요.
        """

        try:
            # 최신 API 호출 방식
            response = client.chat.completions.create(
                model="gpt-4",  # 필요에 따라 gpt-3.5-turbo로 변경 가능
                messages=[
                    {"role": "system", "content": "당신은 서울의 젠트리피케이션과 상권 변화를 분석하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            generated_text = response.choices[0].message.content.strip()

            # 🔍 평가 수행
            factual_score = evaluate_factual_accuracy(generated_text, dong_info)
            consistency_score = evaluate_consistency(generated_text, cluster)

            print(f"🔍 [시도 {attempt + 1}] 사실성: {factual_score}점 | 일관성: {consistency_score}점")

            # ✅ 기준 충족 시 반환
            if factual_score >= min_score and consistency_score >= min_score:
                return generated_text

        except openai.APIError as e:
            st.error(f"API 호출 실패: {str(e)}")
            return "⚠️ OpenAI API 호출에 실패했습니다."

    # ❌ 기준 미달 시 실패 메시지 반환
    return f"⚠️ '{dong_name}'의 설명을 {max_attempts}번 시도했으나, 사실성({min_score}점)과 일관성({min_score}점) 기준을 만족하지 못했습니다."

