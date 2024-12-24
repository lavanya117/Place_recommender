# Importing Libraries
import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib
from IPython.display import display, HTML
import seaborn as sns
import matplotlib.pyplot as plt

# Load Cleaned Dataset and Model
df = pd.read_csv("cleaned_trip_data.csv")
model = joblib.load("trip_recommendation_model_final.pkl")

# Streamlit App Title
st.title("Trip Recommender System 🚗")

# Sidebar Inputs
st.sidebar.header("User Preferences")
user_city = st.sidebar.text_input("Enter the City").strip()
user_rating = st.sidebar.slider("Preferred Minimum Rating", 0.0, 5.0, 4.0)
user_distance = st.sidebar.number_input("Maximum Distance (in km)", min_value=0.0, value=50.0)
user_duration = st.sidebar.number_input("Maximum Exploration Time (in hours)", min_value=0.0, value=5.0)

# Recommendation Function
def recommend_trip(city, rating, distance, duration):
    city_data = df[df["City"].str.lower() == city.lower()]

    if city_data.empty:
        st.warning(f"Sorry, we don't have recommendations for the city '{city}'.")
        return

    filtered_data = city_data[city_data["Ratings"] >= rating]

    if filtered_data.empty:
        st.warning(f"No recommendations in {city} with a rating of {rating} or higher.")
        return

    features = filtered_data[["Ratings", "Distance(Km)", "Time Duration"]].values
    model.fit(features)

    query = [[rating, distance, duration]]
    distances, indices = model.kneighbors(query)

    recommendations = filtered_data.iloc[indices[0]]

    st.subheader(f"Recommended Places and Activities for {city} 🚗")
    for _, row in recommendations.iterrows():
        st.write(f"- **{row['Place']}** (Rating: {row['Ratings']})")
        st.image(row["Images"], width=300)

# Show Recommendations
if st.sidebar.button("Show Recommendations"):
    if user_city:
        recommend_trip(user_city, user_rating, user_distance, user_duration)
    else:
        st.warning("Please enter a city to get recommendations.")

# Data Visualization Section
st.header("Data Insights")
if st.checkbox("Show Boxplot of Ratings by City"):
    st.write("**Boxplot of Ratings by City**")
    plt.figure(figsize=(17, 10))
    sns.boxplot(x="City", y="Ratings", data=df, palette="Set2")
    plt.xticks(rotation=90)
    st.pyplot(plt)

if st.checkbox("Show Dataset"):
    st.write("Cleaned Dataset")
    st.dataframe(df)

# Footer
st.markdown("---")

