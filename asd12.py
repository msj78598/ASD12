import streamlit as st
import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Paths
model_path = r"ASD12_XGBoost.pkl"
template_path = r"C:\ASD12\The data frame file to be analyzed.xlsx"
metrics_path = r"C:\ASD12\metrics.txt"

# Save Metrics
def save_metrics_to_file(accuracy, precision, recall, f1):
    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write("### Model Performance Metrics:\n")
        file.write(f"Accuracy: {accuracy:.2f}\n")
        file.write(f"Precision: {precision:.2f}\n")
        file.write(f"Recall: {recall:.2f}\n")
        file.write(f"F1 Score: {f1:.2f}\n")

# Train Model
def train_and_save_model():
    data_path = r"C:\ASD12\final_classified_loss_with_reasons_60_percent_ordered.xlsx"
    data = pd.read_excel(data_path)
    X = data[['V1', 'V2', 'V3', 'A1', 'A2', 'A3']]
    y = data['Loss_Status'].apply(lambda x: 1 if x == 'Loss' else 0)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    save_metrics_to_file(accuracy, precision, recall, f1)

# Add Loss Reason
def add_loss_reason(row):
    if row['V1'] == 0 and row['A1'] > 0:
        return 'فاقد بسبب وجود تيار مع غياب الجهد على الطور الأول'
    elif row['V2'] == 0 and row['A2'] > 0:
        return 'فاقد بسبب وجود تيار مع غياب الجهد على الطور الثاني'
    elif row['V3'] == 0 and row['A3'] > 0:
        return 'فاقد بسبب وجود تيار مع غياب الجهد على الطور الثالث'
    elif row['V1'] < 10 and row['A1'] > 0:
        return 'فاقد بسبب انخفاض الجهد عن الحد الطبيعي مع وجود تيار على الطور الأول'
    elif row['V2'] < 10 and row['A2'] > 0:
        return 'فاقد بسبب انخفاض الجهد عن الحد الطبيعي مع وجود تيار على الطور الثاني'
    elif row['V3'] < 10 and row['A3'] > 0:
        return 'فاقد بسبب انخفاض الجهد عن الحد الطبيعي مع وجود تيار على الطور الثالث'
    elif abs(row['A1'] - row['A2']) > 0.6 * max(row['A1'], row['A2']) and row['A3'] == 0:
        return 'فاقد بسبب وجود اختلاف كبير بين التيارات على الطور الأول والثاني مع غياب التيار على الطور الثالث'
    elif row['V1'] == 0 and row['A1'] == 0 and abs(row['A2'] - row['A3']) > 0.6 * max(row['A2'], row['A3']):
        return 'فاقد بسبب غياب الجهد والتيار على الطور الأول مع فرق كبير بين الطورين الثاني والثالث'
    elif row.get('CT_Status') == 'Open':
        return 'فاقد بسبب فتح دائرة التيار (CT Open)'
    elif row.get('Line_Status') == 'Failure':
        return 'فاقد بسبب عطل في الخط الكهربائي'
    elif row.get('Read_Status') == 'Missing':
        return 'فاقد بسبب عدم وجود قراءة للعداد'
    else:
        return 'اسباب أخرى للفاقد المحتمل'

# Analyze Data
def analyze_data(data, model):
    try:
        X = data[['V1', 'V2', 'V3', 'A1', 'A2', 'A3']]
        predictions = model.predict(X)
        data['Predicted_Loss'] = predictions
        data['Reason'] = data.apply(add_loss_reason, axis=1)
        
        # Filter only loss cases
        loss_data = data[data['Predicted_Loss'] == 1]
        
        st.write(f"### Number of Detected Loss Cases: {len(loss_data)}")
        st.dataframe(loss_data)
        
        output_path = r"C:\ASD12\analyzed_data_loss_only.xlsx"
        loss_data.to_excel(output_path, index=False)
        
        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Loss Cases",
                data=file,
                file_name="loss_cases.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    except Exception as e:
        st.error(f"Error analyzing data: {e}")

# Streamlit UI
st.title("Energy Loss Prediction System with XGBoost")

st.write("Download the data template:")
with open(template_path, "rb") as file:
    st.download_button(
        label="Download Template",
        data=file,
        file_name="The data frame file to be analyzed.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


uploaded_file = st.file_uploader("Upload data for analysis (Excel)", type=["xlsx"])
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    if not os.path.exists(model_path):
        train_and_save_model()
    model = joblib.load(model_path)
    analyze_data(data, model)


st.title("المطور / مشهور العباس ")
