import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix

# โหลดโมเดล Machine Learning
svm_air_quality_model = joblib.load('air_quality_svm_model.pkl')
air_quality_scaler = joblib.load('air_quality_scaler.pkl')

# โหลดโมเดล Neural Network
mlp_fruit_model = joblib.load('fruit_mlp_model.pkl')
fruit_scaler = joblib.load('fruit_scaler.pkl')
fruit_label_encoder = joblib.load('fruit_label_encoder.pkl')

# สร้างแอป Streamlit
st.title("Machine Learning & Neural Network Web Application")

# เมนูนำทาง
menu = ["Project Overview", "Air Quality Prediction (ML)", "Fruit Classification (NN)", "Batch Prediction", "Model Evaluation"]
choice = st.sidebar.selectbox("Select Page", menu)

# หน้าสำหรับอธิบายโปรเจกต์
if choice == "Project Overview":
    st.header("Project Overview")
    st.write("""
        This project demonstrates two models:
        1. **Machine Learning Model (SVM)** for Air Quality Prediction.
        2. **Neural Network Model (MLPClassifier)** for Fruit Classification.
        3. **Batch Prediction** for uploading CSV files.
        4. **Model Evaluation** for assessing model performance.
    """)

# ฟังก์ชันสำหรับ input แบบทั้งกรอกและเลื่อนค่าได้
def hybrid_input(label, min_value, max_value, default):
    return st.number_input(label, min_value=min_value, max_value=max_value, value=default, step=0.1, format="%.2f")

# หน้าสำหรับทำนายคุณภาพอากาศด้วย Machine Learning
if choice == "Air Quality Prediction (ML)":
    st.header("Air Quality Prediction (Machine Learning)")

    # ข้อมูลตัวอย่างสำหรับแต่ละระดับ
    example_options = {
        "Good": {"PM2.5": 5.0, "PM10": 20.0, "Temperature": 22.0, "Humidity": 50.0, "Wind Speed": 10.0},
        "Moderate": {"PM2.5": 40.0, "PM10": 70.0, "Temperature": 28.0, "Humidity": 40.0, "Wind Speed": 5.0},
        "Poor": {"PM2.5": 100.0, "PM10": 150.0, "Temperature": 35.0, "Humidity": 30.0, "Wind Speed": 2.0}
    }

    selected_example = st.selectbox("Select Example Data", ["Custom", "Good", "Moderate", "Poor"])
    example_data_air_quality = example_options.get(selected_example, {"PM2.5": 0.0, "PM10": 0.0, "Temperature": 0.0, "Humidity": 0.0, "Wind Speed": 0.0})

    pm25 = hybrid_input("PM2.5", 0.0, 200.0, float(example_data_air_quality["PM2.5"]))
    pm10 = hybrid_input("PM10", 0.0, 300.0, float(example_data_air_quality["PM10"]))
    temp = hybrid_input("Temperature (C)", -50.0, 50.0, float(example_data_air_quality["Temperature"]))
    humidity = hybrid_input("Humidity (%)", 0.0, 100.0, float(example_data_air_quality["Humidity"]))
    wind_speed = hybrid_input("Wind Speed (km/h)", 0.0, 50.0, float(example_data_air_quality["Wind Speed"]))

    if st.button("Predict Air Quality"):
        with st.spinner('Predicting...'):
            input_data = air_quality_scaler.transform(np.array([[pm25, pm10, temp, humidity, wind_speed]]))
            prediction = svm_air_quality_model.predict(input_data)[0]
            probabilities = svm_air_quality_model.predict_proba(input_data)[0]

            quality = {0: "Good", 1: "Moderate", 2: "Poor"}
            st.success(f"Predicted Air Quality: {quality[prediction]}")

            # แสดงผลการทำนายแบบความน่าจะเป็นด้วย Plotly
            prob_df = pd.DataFrame({'Air Quality': quality.values(), 'Probability': probabilities})
            fig = px.bar(prob_df, x='Air Quality', y='Probability', color='Air Quality', title='Prediction Confidence')
            st.plotly_chart(fig)

            # แสดงกราฟเปรียบเทียบค่าปัจจัยกับมาตรฐานด้วย Plotly
            factors = ['PM2.5', 'PM10', 'Temperature', 'Humidity', 'Wind Speed']
            values = [pm25, pm10, temp, humidity, wind_speed]
            standards = [12, 50, 25, 50, 10]  # ค่ามาตรฐานอากาศดี

            comparison_df = pd.DataFrame({'Factors': factors, 'Your Input': values, 'Good Standard': standards})
            fig_comparison = px.line(comparison_df, x='Factors', y=['Your Input', 'Good Standard'], markers=True, title='Air Quality Factors Comparison')
            st.plotly_chart(fig_comparison)

            # บันทึกผลลัพธ์เป็นไฟล์ CSV
            output_df = pd.DataFrame({'PM2.5': [pm25], 'PM10': [pm10], 'Temperature': [temp], 'Humidity': [humidity], 'Wind Speed': [wind_speed], 'Predicted Air Quality': [quality[prediction]]})
            csv = output_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Prediction Results", data=csv, file_name='air_quality_prediction.csv', mime='text/csv')

# หน้าสำหรับจำแนกผลไม้ด้วย Neural Network
if choice == "Fruit Classification (NN)":
    st.header("Fruit Classification (Neural Network)")

    # ข้อมูลตัวอย่างสำหรับผลไม้แต่ละประเภท
    fruit_examples = {
        "Apple": {"Weight": 180.0, "Length": 8.0, "Circumference": 28.0, "Color": 0},
        "Banana": {"Weight": 120.0, "Length": 18.0, "Circumference": 12.0, "Color": 1},
        "Orange": {"Weight": 220.0, "Length": 7.0, "Circumference": 23.0, "Color": 2}
    }

    selected_fruit_example = st.selectbox("Select Example Fruit", ["Custom", "Apple", "Banana", "Orange"])
    example_data_fruit = fruit_examples.get(selected_fruit_example, {"Weight": 0.0, "Length": 0.0, "Circumference": 0.0, "Color": 0})

    weight = hybrid_input("Weight (g)", 0.0, 500.0, float(example_data_fruit["Weight"]))
    length = hybrid_input("Length (cm)", 0.0, 30.0, float(example_data_fruit["Length"]))
    circumference = hybrid_input("Circumference (cm)", 0.0, 40.0, float(example_data_fruit["Circumference"]))
    color = st.selectbox("Color", [0, 1, 2], format_func=lambda x: ["Green", "Yellow", "Orange"][x], index=int(example_data_fruit["Color"]))

    if st.button("Classify Fruit"):
        with st.spinner('Classifying...'):
            input_data = fruit_scaler.transform(np.array([[weight, length, circumference, color]]))
            prediction = mlp_fruit_model.predict(input_data)[0]
            fruit_name = fruit_label_encoder.inverse_transform([prediction])[0]
            st.success(f"Predicted Fruit Type: {fruit_name}")

            # แสดงกราฟความมั่นใจของโมเดลด้วย Plotly
            probabilities = mlp_fruit_model.predict_proba(input_data)[0]
            prob_df_fruit = pd.DataFrame({'Fruit Type': fruit_label_encoder.classes_, 'Probability': probabilities})
            fig_fruit = px.bar(prob_df_fruit, x='Fruit Type', y='Probability', color='Fruit Type', title='Prediction Confidence')
            st.plotly_chart(fig_fruit)

            # บันทึกผลลัพธ์เป็นไฟล์ CSV
            output_df_fruit = pd.DataFrame({'Weight': [weight], 'Length': [length], 'Circumference': [circumference], 'Color': [color], 'Predicted Fruit Type': [fruit_name]})
            csv_fruit = output_df_fruit.to_csv(index=False).encode('utf-8')
            st.download_button("Download Classification Results", data=csv_fruit, file_name='fruit_classification.csv', mime='text/csv')

# ฟังก์ชันสำหรับอัพโหลดและทำนายแบบแบทช์
if choice == "Batch Prediction":
    st.header("Batch Prediction with CSV Upload")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(df)

        if 'PM2.5' in df.columns and 'PM10' in df.columns and 'Temperature' in df.columns and 'Humidity' in df.columns and 'Wind_Speed' in df.columns:
            st.write("Performing Air Quality Prediction...")
            input_data = air_quality_scaler.transform(df[['PM2.5', 'PM10', 'Temperature', 'Humidity', 'Wind_Speed']])
            predictions = svm_air_quality_model.predict(input_data)
            df['Predicted_Air_Quality'] = ["Good" if pred == 0 else "Moderate" if pred == 1 else "Poor" for pred in predictions]
            st.write("Predicted Results:")
            st.table(df)

            # บันทึกผลลัพธ์เป็นไฟล์ CSV
            csv_batch_air_quality = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Batch Prediction Results", data=csv_batch_air_quality, file_name='batch_air_quality_predictions.csv', mime='text/csv')

        elif 'Weight' in df.columns and 'Length' in df.columns and 'Circumference' in df.columns and 'Color' in df.columns:
            st.write("Performing Fruit Classification...")
            input_data = fruit_scaler.transform(df[['Weight', 'Length', 'Circumference', 'Color']])
            predictions = mlp_fruit_model.predict(input_data)
            df['Predicted_Fruit_Type'] = fruit_label_encoder.inverse_transform(predictions)
            st.write("Predicted Results:")
            st.table(df)

            # บันทึกผลลัพธ์เป็นไฟล์ CSV
            csv_batch_fruit = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Batch Classification Results", data=csv_batch_fruit, file_name='batch_fruit_classifications.csv', mime='text/csv')

        else:
            st.error("The uploaded file does not have the required columns for prediction.")

# หน้าสำหรับประเมินผลโมเดล
if choice == "Model Evaluation":
    st.header("Model Evaluation")

    # อัปโหลดไฟล์สำหรับประเมินผลโมเดล
    uploaded_file_eval = st.file_uploader("Upload CSV for Model Evaluation", type=["csv"])
    if uploaded_file_eval is not None:
        df_eval = pd.read_csv(uploaded_file_eval)
        st.write("Uploaded Data for Evaluation:")
        st.dataframe(df_eval)

        if 'PM2.5' in df_eval.columns and 'PM10' in df_eval.columns and 'Temperature' in df_eval.columns and 'Humidity' in df_eval.columns and 'Wind_Speed' in df_eval.columns and 'Actual_Air_Quality' in df_eval.columns:
            st.write("Evaluating Air Quality Prediction Model...")
            input_data = air_quality_scaler.transform(df_eval[['PM2.5', 'PM10', 'Temperature', 'Humidity', 'Wind_Speed']])
            predictions = svm_air_quality_model.predict(input_data)
            report = classification_report(df_eval['Actual_Air_Quality'], predictions, output_dict=True)
            st.write(pd.DataFrame(report).transpose())

            conf_matrix = confusion_matrix(df_eval['Actual_Air_Quality'], predictions)
            st.write("Confusion Matrix:")
            st.write(conf_matrix)

        elif 'Weight' in df_eval.columns and 'Length' in df_eval.columns and 'Circumference' in df_eval.columns and 'Color' in df_eval.columns and 'Actual_Fruit_Type' in df_eval.columns:
            st.write("Evaluating Fruit Classification Model...")
            input_data = fruit_scaler.transform(df_eval[['Weight', 'Length', 'Circumference', 'Color']])
            predictions = mlp_fruit_model.predict(input_data)
            report = classification_report(fruit_label_encoder.transform(df_eval['Actual_Fruit_Type']), predictions, output_dict=True)
            st.write(pd.DataFrame(report).transpose())

            conf_matrix = confusion_matrix(fruit_label_encoder.transform(df_eval['Actual_Fruit_Type']), predictions)
            st.write("Confusion Matrix:")
            st.write(conf_matrix)

        else:
            st.error("The uploaded file does not have the required columns for evaluation.")
