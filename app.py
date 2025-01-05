import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Configure the Streamlit page
st.set_page_config(
    page_title="Graduate Admission Predictor",
    page_icon="🎓",
    layout="centered"
)

def load_model():
    try:
        # Attempt to load the trained model
        model = pickle.load(open('admission_model.pkl', 'rb'))
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_admission(model, features):
    try:
        # Make prediction using the model
        prediction = model.predict([features])[0]
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def main():
    # Add title and description
    st.title("Graduate Admission Chance Predictor 🎓")
    st.write("Enter your academic details to predict admission chances")
    
    # Create input form
    with st.form("prediction_form"):
        # Input fields
        gre = st.slider("GRE Score", 260, 340, 300)
        toefl = st.slider("TOEFL Score", 0, 120, 100)
        university_rating = st.slider("University Rating", 1, 5, 3)
        sop = st.slider("Statement of Purpose (SOP) Rating", 1.0, 5.0, 3.0, 0.5)
        lor = st.slider("Letter of Recommendation (LOR) Rating", 1.0, 5.0, 3.0, 0.5)
        cgpa = st.slider("CGPA", 6.0, 10.0, 8.0, 0.1)
        research = st.radio("Research Experience", ["Yes", "No"])
        
        # Submit button
        submitted = st.form_submit_button("Predict Admission Chance")
    
    if submitted:
        # Load the model
        model = load_model()
        
        if model:
            # Prepare input features
            research_int = 1 if research == "Yes" else 0
            features = [gre, toefl, university_rating, sop, lor, cgpa, research_int]
            
            # Get prediction
            prediction = predict_admission(model, features)
            
            if prediction is not None:
                # Display results
                st.success(f"Your predicted chance of admission is {prediction*100:.2f}%")
                
                # Show interpretation
                if prediction >= 0.8:
                    st.write("🌟 Excellent profile! Very strong chances of admission.")
                elif prediction >= 0.6:
                    st.write("👍 Good profile! Decent chances of admission.")
                else:
                    st.write("💪 Consider strengthening your profile or applying to more universities.")
                
                # Show feature importance disclaimer
                st.info("Note: GRE, TOEFL, and CGPA typically have the strongest impact on admission chances.")

    # Add footer with additional information
    st.markdown("---")
    st.markdown("### How to improve your chances:")
    st.write("""
    - Focus on maintaining a high CGPA
    - Prepare well for GRE and TOEFL
    - Try to gain research experience
    - Get strong letters of recommendation
    """)

if __name__ == "__main__":
    main()