import streamlit as st

# Import necessary functions from web_functions
from web_function import load_data

from Tab import Home, data, detection, prediction

# Configure the app
st.set_page_config(
    page_title='Student Stress Detector',
    page_icon=':worried:',
    layout='wide',
    initial_sidebar_state='auto'
)

# You can customize the colour of output page by going to settings
# Dictionary for pages

Tabs = {
    "Home": Home,
    "Data Info": data,
    "Detection": detection,
    "Prediction": prediction,
}

# Create a sidebar
# Add title to sidebar
st.sidebar.title("Navigation")

# Create radio option to select the page
page = st.sidebar.radio("Pages", list(Tabs.keys()))

# Loading the dataset.
df, x, y = load_data()

# Call the app function of selected page to run
if page in ["Prediction","Detection"]:
    Tabs[page].app(df, x, y)
elif page == "Data Info":
    Tabs[page].app(df)
else:
    Tabs[page].app()   # type: ignore