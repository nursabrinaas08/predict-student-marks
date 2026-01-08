import streamlit as st
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Marks Prediction",page_icon="🧑🏾‍🎓",layout="centered")

st.title("💯👩🏾‍🎓Student Marks Predictor")
st.write("Enter The Number Of Hours Studied (1-10) And Click **Predict** To See The Predicted Marks Based On Your Hour Of Study")

# Load The Model (Pickle file)
def load_model(model):  # parameterize function
  with open(model,'rb') as f:
    slr = pickle.load(f)
  return slr

try:
  model = load_model("slr.pkl")
except Exception as e:
  st.error("Your pickle file not found")
  st.exception("Failed to load the model :",e)
  st.stop()


hours = st.number_input("Hours Studied",
                        min_value=1.0,
                        max_value=10.0,
                        value=1.0,
                        step=0.1,
                        format="%.1f")

if st.button("Predict"):
  try:
    X = np.array([[hours]])
    predict = model.predict(X)
    predict = predict[0]
    
    st.success(f"Predicted Marks : {round(predict,1)}")
    st.write("Note : This is ML Model Prediction. Result May Vary")
  except Exception as e:
    st.error("Prediction failed :",e) 
