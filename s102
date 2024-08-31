import streamlit as st
import numpy as np
import joblib

# 모델과 스케일러 로드
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit 앱 시작
st.title("당뇨병 여부 진단기")

# 사용자 입력
pregnancies = st.number_input("임신 횟수", min_value=0, max_value=20, value=0)
glucose = st.number_input("공복 혈당", min_value=0, max_value=200, value=0)
blood_pressure = st.number_input("혈압", min_value=0, max_value=150, value=0)
skin_thickness = st.number_input("피부 두께", min_value=0, max_value=100, value=0)
insulin = st.number_input("인슐린", min_value=0, max_value=1000, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=0.0)
diabetes_pedigree = st.number_input("당뇨병 가족력", min_value=0.0, max_value=3.0, value=0.0)
age = st.number_input("나이", min_value=0, max_value=120, value=0)

# 입력값을 데이터프레임으로 변환
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                        insulin, bmi, diabetes_pedigree, age]])

# 입력 데이터 정규화
input_data_scaled = scaler.transform(input_data)

# 예측 버튼
if st.button("예측"):
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 1:
        st.success("당뇨병 발병 가능성이 있습니다.")
    else:
        st.success("당뇨병 발병 가능성이 없습니다.")
