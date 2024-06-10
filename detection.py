import warnings
import matplotlib.pyplot as plt
import seaborn as sns

'''from sklearn.metrics import plot_confusion_matrix'''
from sklearn import tree
import streamlit as st

# Import necessary functions from web_functions
from web_function import train_model


def app(df, x, y):
    """This function create the visualisation page"""

    # Remove the warnings
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Set the page title
    st.title("Visualise the Stress Level")

    # Create a checkbox to show correlation heatmap
    if st.checkbox("Show the correlation heatmap"):
        st.subheader("Correlation Heatmap")

        fig = plt.figure(figsize=(10, 6))
        ax = sns.heatmap(df.iloc[:, 1:].corr(),annot=True)  # Creating an object of seaborn axis and storing it in 'ax' variable
        bottom, top = ax.get_ylim()  # Getting the top and bottom margin limits.
        ax.set_ylim(bottom + 0.5, top - 0.5)  # Increasing the bottom and decreasing the top margins respectively.
        st.pyplot(fig)

    if st.checkbox("Show Scatter Plot"):
        figure, axis = plt.subplots(2, 2, figsize=(15, 10))

        sns.scatterplot(ax=axis[0, 0], data=df, x='blood_pressure', y='headache')
        axis[0, 0].set_title("Blood Pressure vs Headache")

        sns.scatterplot(ax=axis[0, 1], data=df, x='sleep_quality', y='breathing_problem')
        axis[0, 1].set_title("sleep_quality vs breathing_problem")

        sns.scatterplot(ax=axis[1, 0], data=df, x='study_load', y='teacher_student_relationship')
        axis[1, 0].set_title("Study Load vs Teacher Student Relationship")

        sns.scatterplot(ax=axis[1, 1], data=df, x='sleep_quality', y='blood_pressure')
        axis[1, 1].set_title("Sleeping Quality vs Blood Pressure")
        st.pyplot()

    if st.checkbox("Display Boxplot"):
        fig, ax = plt.subplots(figsize=(15, 5))
        df.boxplot(['blood_pressure', 'headache', 'sleep_quality', 'breathing_problem', 'study_load', 'teacher_student_relationship'], ax=ax)
        st.pyplot()

    if st.checkbox("Show Sample Results"):
        safe = (df['bullying'] == 0).sum()
        low = (df['bullying'] == 1).sum()
        med = (df['bullying'] == 2).sum()
        high = (df['bullying'] == 3).sum()
        vhigh = (df['bullying'] == 4).sum()
        data = [safe, low, med, high, vhigh]
        labels = ['Safe', 'Low', 'Medium', 'High', 'Very High']
        colors = sns.color_palette('pastel')[0:7]
        plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')
        st.pyplot()




