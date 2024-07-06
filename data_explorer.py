import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


# Column names for the dataset
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
# Read the dataset
def load_data():
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header=None, names=column_names)
    return df

data = load_data()

# Title and introduction
st.title("Exploring Iris Data")

# Display raw data
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.dataframe(data)

# Show the average sepal length for each species
if st.checkbox("Show the average sepal length for each species"):
    st.subheader("Average Sepal Length per Species")
    avg_sepal_length = data.groupby("class")["sepal_length"].mean()
    st.write(avg_sepal_length)

# Display a scatter plot comparing two features
    #Specific features
#if st.checkbox("Show a scatter plot of sepal width and length"):
st.subheader("Scatter Plot of Sepal Width and Length")
fig = px.scatter(data, x='sepal_width', y='sepal_length', color='class', title='Scatter Plot of Iris Data')
st.plotly_chart(fig)

#Plot of any two features
st.subheader("Compare two features using a scatter plot")
feature_1 = st.selectbox("Select the first feature:", data.columns[:-1])
feature_2 = st.selectbox("Select the second feature:", data.columns[:-1])
fig = px.scatter(data, x=feature_1, y=feature_2, color='class', title='Scatter Plot of Iris Data')
st.plotly_chart(fig)

# Filter data based on species
species = data['class'].unique()
species_filtered = st.multiselect("Select species:",species, default=species)
if species_filtered:
    filtered_data = data[data["class"].isin(species_filtered)]
    st.dataframe(filtered_data)
else:
    st.write("No species selected.")

#Display a pairplot for the selected species
if st.checkbox("Show pairplot for the selected species"):
    st.subheader("Pairplot for the Selected Species")

    if species_filtered:
        sns.pairplot(filtered_data, hue="class")
    else:
        sns.pairplot(data, hue="class")
        
    st.pyplot()

# Question 5: Show the distribution of a selected feature
st.subheader("Distribution of a Selected Feature")
selected_feature = st.selectbox("Select a feature to display its distribution:", data.columns[:-1])

hist_plot = px.histogram(data, x=selected_feature, color="class", nbins=30, marginal="box", hover_data=data.columns)
st.plotly_chart(hist_plot)