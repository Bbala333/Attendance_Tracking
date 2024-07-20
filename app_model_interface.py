import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('attendance_model.pkl')

# Function to get manual input from user
def get_manual_input():
    previous_day_attendance = st.radio("Previous Day Attendance", options=['Absent', 'Present'], index=1)  # Radio buttons for Present and Absent
    total_workdays = st.number_input("Total Workdays up to Today", min_value=0, value=0, step=1)
    total_attended = st.number_input("Total Attended Days up to Today", min_value=0, value=0, step=1)
    
    return 1 if previous_day_attendance == 'Present' else 0, total_workdays, total_attended

# Streamlit app
st.title("Employee Attendance Prediction")

st.write("""
## Predict whether the employee will be present based on past attendance data.
""")

# Background GIF
page_bg_img = f"""                  
<style>               
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://cdn.dribbble.com/users/1308090/screenshots/8808655/media/2b6ff6ba4393394c9e41295c5a3c329c.gif");
background-position: top;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
.block-container {{
background-color: rgb(121 179 112 / 60%);;
padding: 20px;
border-radius: 10px;
}}.st-emotion-cache-eqffof {{
    font-family: "Source Sans Pro", sans-serif;
    margin-bottom: -1rem;
    background: aliceblue;
    border-radius: 5px;
    padding-left: 11px;
    margin-bottom:4px;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Get input data from user
previous_day_attendance, total_workdays, total_attended = get_manual_input()

# Validate input
if total_workdays < total_attended:
    st.warning("Total Workdays should be greater than or equal to Total Attended Days.")
else:
    input_data = pd.DataFrame({
        'Previous_Day_Attendance': [previous_day_attendance],
        'Total_Workdays': [total_workdays],
        'Total_Attended': [total_attended]
    })

    # Predict using the model
    if st.button("Predict"):
        prediction = model.predict(input_data)
        
        # Check if the model supports predict_proba
        if hasattr(model, "predict_proba"):
            prediction_prob = model.predict_proba(input_data)[:, 1]  # Get the probability of class 1 (Present)
            
            st.markdown('<div class="transparent-card">', unsafe_allow_html=True)
            st.write(f"**Prediction Probability:** {prediction_prob[0]:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        attendance_status = "Present" if prediction[0] == 1 else "Absent"
        st.markdown('<div class="transparent-card">', unsafe_allow_html=True)
        st.write(f"**Predicted Attendance Status:** {attendance_status}")
        st.markdown('</div>', unsafe_allow_html=True)

# To run the Streamlit app, use the command: streamlit run app.py
