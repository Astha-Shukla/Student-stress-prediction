import streamlit as st

# Import necessary functions from web_functions
from web_function import predict


def app(df, x, y):
    """This function create the prediction page"""

    # Add title to the page
    st.title("Prediction Page")

    # Add a brief description
    st.markdown(
        """
            <p style="font-size:25px">
                This app uses <b style="color:green">Decision Tree Classifier</b> for the Prediction of Stress Level.
            </p>
        """, unsafe_allow_html=True)

    # Take feature input from the user
    # Add a subheader
    st.subheader("Select Values:")

    # Take input of features from the user.
    blood_pressure = st.slider("Blood Pressure", int(df["blood_pressure"].min()), int(df["blood_pressure"].max()))
    headache = st.slider("Headache", int(df["headache"].min()), int(df["headache"].max()))
    anxiety_level = st.slider("Anxiety Level", int(df["anxiety_level"].min()), int(df["anxiety_level"].max()))
    depression  = st.slider("Depression", float(df["depression"].min()), float(df["depression"].max()))
    social_support = st.slider("Social Support", float(df["social_support"].min()), float(df["social_support"].max()))
    #stress_level = st.slider("Stress Level", float(df["stress_level"].min()), float(df["stress_level"].max()))

    # Create a list to store all the features
    features = [blood_pressure, headache, anxiety_level, depression, social_support]

    # Create a button to predict
    if st.button("Predict"):
        # Get prediction and model score
        prediction, score = predict(x, y, features)
        st.info("Stress level detected...")

        # Print the output according to the prediction
        if (prediction == 1):
            st.success("The person has low stress level 🙂")
            st.write("Stress score =", str(prediction[0]))
        elif (prediction == 2):
            st.warning("The person has medium stress level 😐")
            st.write("Stress score =", str(prediction[0]))
        elif (prediction == 3):
            st.error("The person has high stress level! 😞")
            st.write("Stress score =", str(prediction[0]))
        elif (prediction == 4):
            st.error("The person has very high stress level!! 😫")
            st.write("Stress score =", str(prediction[0]))
        else:
            st.success("The person is stress free and calm 😄")
