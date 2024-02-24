import sys
import streamlit as st
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained SVR model
model_path = 'model_pkl_1.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the scaler from the training phase
scaler_path = 'scaler2.pkl'  # You should save the scaler during training
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Function to scale input features
def scale_features(batch_size, complexity_score, protein_source, vegetables):
    input_data = pd.DataFrame({
        'Batch Size': [batch_size],
        'Complexity Score': [complexity_score],
        'Protein Source_Pork': [0],
        'Protein Source_Beef': [0],
        'Protein Source_Chicken': [0],
        'Protein Source_Salmon': [0],
        'Protein Source_Tofu': [0],
        'Protein Source_Shrimp': [0],
        'Protein Source_Tuna': [0],
        'Protein Source_White Fish': [0],
        'Vegetables_Artichoke': [0],
        'Vegetables_Arugula': [0],
        'Vegetables_Asparagus': [0],
        'Vegetables_Bean': [0],
        'Vegetables_Beet': [0],
        'Vegetables_Brussels Sprouts': [0],
        'Vegetables_Broccoli': [0],
        'Vegetables_Carrots': [0],
        'Vegetables_Cauliflower': [0],
        'Vegetables_Collard Greens': [0],
        'Vegetables_Green Beans': [0],
        'Vegetables_Kale': [0],
        'Vegetables_Mushrooms': [0],
        'Vegetables_Pineapple': [0],
        'Vegetables_Potatoes': [0],
        'Vegetables_Squash': [0],
        'Vegetables_Sweet Potatoes': [0],
        'Vegetables_Zuchini': [0]
    })


    #Set the appropriate one-hot encoded columns if protein_source and vegetables are not None
    if protein_source:
        input_data[f'Protein Source_{protein_source}'] = 1
    if vegetables:
        input_data[f'Vegetables_{vegetables}'] = 1

    scaled_input = scaler.transform(input_data)
    return scaled_input

# Streamlit app
def main():
    st.title("Unit Time Prediction")

    # User input for 'Batch Size', 'Complexity Score', 'Protein Source', and 'Vegetables'
    
    batch_size = st.text_input("Enter Batch Size:", 0.0)
    
    
    complexity_score = st.text_input("Enter Complexity Score:", 0.0)
    
    
    protein_source = st.selectbox("Select Primary Protein Source:", ["", "Pork", "Beef", "Chicken", "Salmon", "Tofu", "Shrimp", "Tuna", "White Fish"])
    
    
    vegetables = st.selectbox("Select Primary Vegetable/Fruit:", ["", "Artichoke", "Arugula", "Asparagus", "Bean", "Beet", "Brussels Sprouts", "Broccoli", "Carrots", "Cauliflower", "Collard Greens", "Green Beans", "Kale", "Mushrooms", "Pineapple", "Potatoes", "Squash", "Sweet Potatoes", "Zuchini"])

    # Convert input values to float (or handle non-numeric inputs gracefully)
    try:
        batch_size = float(batch_size)
        complexity_score = float(complexity_score)
    except ValueError:
        st.error("Please enter valid numeric values for Batch Size and Complexity Score.")
        return

    # Scale input features
    scaled_input = scale_features(batch_size, complexity_score, protein_source, vegetables)

    # Make prediction
    prediction = model.predict(scaled_input)
    
    # Calculate estimated total time
    estimated_total_time = batch_size * prediction[0]

    st.subheader("Prediction:")
    st.write(f"Predicted Time: {prediction[0]:.2f} minutes/lb")
    st.write(f"Estimated Total Time: {estimated_total_time:.2f} minutes")

if __name__ == "__main__":
    # Run the Streamlit app
    main()