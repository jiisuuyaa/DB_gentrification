import os
import joblib
import numpy as np
from dotenv import load_dotenv
import openai
import xgboost as xgb
import pandas as pd


# 📂 .env 파일에서 환경 변수 로드
load_dotenv()

# 🔑 OpenAI API 키 불러오기
openai_api_key = os.getenv("OPENAI_API_KEY")

# 🔹 OpenAI 클라이언트 설정
client = openai.OpenAI(api_key=openai_api_key)

# 📂 XGBoost 모델 불러오기
xgb_clf  = joblib.load("xgboost_model.pkl")
encoder  = joblib.load("label_encoder.pkl")  # LabelEncoder 로드


# 위험 단계에 대한 설명 맵핑
risk_description_map = {
    "위험_상권발달형": "🔴 이 지역은 상권이 발전했지만 SNS 활동이 적고 젠트리피케이션이 상대적으로 덜 진행된 지역입니다. 비슷한 서울 지역으로는 노량진, 대치동 등이 있습니다.",
    "위험_관심집중형": "🔴 이 지역은 SNS에서 큰 관심을 받고 있으며, 젊은 층이 많이 방문해 젠트리피케이션이 빠르게 진행 중입니다. 비슷한 서울 지역으로는 명동, 연남동 등이 있습니다",
    "위험_균형진행형": "🔴 이 지역은 상권, SNS 활동, 젠트리피케이션이 균형을 이루는 지역으로 점진적 변화가 예상됩니다. 비슷한 서울 지역으로는 상암동, 독산동 등이 있습니다.",
    "위험_젠트리피케이션과열형": "🔴 이 지역은 젠트리피케이션이 과열되었으나, 상권 활성화는 상대적으로 낮은 지역입니다. 비슷한 서울 지역으로는 신도림, 아현동 등이 있습니다.",
    "경계": "🟠 이 지역은 상권이 빠르게 성장 중이며, 젠트리피케이션 진행 가능성이 높은 지역입니다.",
    "주의": "🟡 이 지역은 상권이 서서히 성장하는 지역이며, 일부 지역에서 젠트리피케이션 변화가 시작될 가능성이 있습니다.",
    "일반": "🟢 이 지역은 상권이 안정적이며 젠트리피케이션 영향이 적은 지역으로, 임대료 상승 등의 변화도 미미한 편입니다."
}

# 🔍 1️⃣ 입력된 지수로 위험 단계 예측
def predict_risk(gentrification_index, business_activation_index, gentrification_sns_index, business_activation_sns_index):
    # 🚀 컬럼 순서를 맞춘 DataFrame으로 변환
    train_columns = ['상권활성화지수', '상권활성화_SNS', '젠트리피케이션지수', '젠트리피케이션_SNS']
    features = pd.DataFrame([[gentrification_index, business_activation_index, gentrification_sns_index, business_activation_sns_index]], 
                            columns=train_columns)

    # 예측 수행
    risk_encoded = xgb_clf.predict(features)[0]
    
    # 🚀 예측된 숫자를 한글 라벨로 변환 (디코딩)
    risk_level = encoder.inverse_transform([risk_encoded])[0]

    # 🔴 위험 단계 추가 설명
    description = ""  # description 변수 초기화
    if "위험_" in risk_level:
        description += "\n⚠️ **이 지역은 다른 지역에 비해 상권 활성화와 젠트리피케이션이 상당히 진행된 상태입니다.**"

    return risk_level, description



# 💡 2️⃣ 생성형 AI로 추가 설명 생성
def generate_ai_risk_description(risk_level, region_name, gentrification_index, business_activation_index, gentrification_sns_index, business_activation_sns_index):
    """
    AI를 사용하여 위험 단계에 대한 심층적이고 맞춤형 설명을 생성하는 함수
    """
    prompt = f"""
    당신은 도시 계획 및 젠트리피케이션 전문가입니다. 
    현재 사용자가 '{region_name}' 지역의 젠트리피케이션 위험을 예측하려고 합니다.

    📢 **💡 젠트리피케이션과 상권 활성화란?**  
    젠트리피케이션(Gentrification)은 특정 지역의 경제가 성장하면서 건물 임대료가 오르고 기존 거주민들이 떠나게 되는 현상을 말합니다.  
    상권 활성화는 소비와 유동 인구가 증가하면서 지역 경제가 활발해지는 것을 뜻합니다.  

    📊 입력된 지수:
    - 🏬 **상권 활성화 지수**: {business_activation_index}
    - 🏙️ **젠트리피케이션 지수**: {gentrification_index}
    - 📢 **SNS 상권 활성화 점수**: {business_activation_sns_index}
    - 💬 **SNS 젠트리피케이션 점수**: {gentrification_sns_index}

    🏙️ AI 모델이 예측한 위험 단계: {risk_level}
    
    🔍 해당 위험 단계에 대한 분석과 미래 전망을 전문가 수준으로 설명해 주세요.
    - 만약 "{risk_level}"에 "위험_"이 포함되어 있으면, 해당 지역은 다른 지역에 비해 상권 활성화와 젠트리피케이션이 상당히 진행된 상태입니다. 이를 고려하여 분석해주세요.
    - 현재 이 지역의 경제적 변화 수준이 다른 지역과 비교했을 때 어느 정도인지 설명해주세요.
    - 상권 활성화가 계속 진행될 가능성이 높은지, 젠트리피케이션이 앞으로 어떤 영향을 미칠지 분석해주세요.
    - 지역 상권과 부동산 시장에 미치는 영향을 포함해서 전망을 이야기해주세요.
    """
    
    # OpenAI API 호출
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # 또는 gpt-4 사용 가능
    messages=[
        {"role": "system", "content": "당신은 젠트리피케이션과 상권 변화를 분석하는 전문가입니다."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2  # 온도를 낮춰서 더 예측 가능한 답변 생성
)

    return response.choices[0].message.content.strip()
