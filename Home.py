import streamlit as st


def app():
    # This function create the home page

    # Add title to the home page
    st.title("Student Stress Detection")

    # Add image to the home page
    st.image("stress.jpg")

    # Add brief description of your web app
    st.markdown(
        """<p style="font-size:20px;"> Stress detector classifies a stressed individual from a normal one by 
        acquiring his/her physiological signals through appropriate sensors such as Headache, 
        Peer Pressure, Blood Pressure, Sleeping Quality, Extra-cirricular Activity etc,. These signals are pre-processed to extract the desired features which 
        depicts the stress level in working individuals. Decision Tree Classifier (DTC) is investigated to classify these extracted feature set. The result 
        indicates feature vector with best features having a strong influence in stress identification. An attempt is 
        made to determine the best feature set that results in maximum classification accuracy. </p>""",
        unsafe_allow_html=True)