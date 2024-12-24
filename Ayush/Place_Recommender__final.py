#!/usr/bin/env python
# coding: utf-8

# **Importing the libraries**
# 

# In[91]:


import pandas as pd
from sklearn.neighbors import NearestNeighbors
from IPython.display import display, HTML
import seaborn as sns
import matplotlib.pyplot as plt


# **Importing the dataset**
# 

# In[6]:


df=pd.read_csv('Trip.csv')


# In[7]:


df.head()


# **Some information about the dataset**
# 

# In[12]:


df.describe()


# In[14]:


df.columns


# **Finding the null values in the dataset**
# 

# In[17]:


df.isnull().sum()


# In[19]:


print(f"There are {df.isnull().sum().sum()} null values in the dataset now. ")


# **Filling up the null values and deleting unnecessary rows**

# In[22]:


df["Ratings"] = df["Ratings"].fillna(round(df["Ratings"].mean(), 2))

df=df.drop(2968)
df=df.drop(81)


# In[24]:


df["Distance(Km)"] = df["Distance(Km)"].astype(str)
df["Distance(Km)"] = df["Distance(Km)"].str.replace(",", "", regex=True)
df["Distance(Km)"] = pd.to_numeric(df["Distance(Km)"], errors="coerce")
df["Distance(Km)"] = df.groupby("City")["Distance(Km)"].transform(lambda x: x.fillna(x.mean()))
df["City"] = df["City"].str.strip()






# In[26]:


df = df.dropna(subset=['Distance(Km)','Time Duration'])



# **After filling up the null values**

# In[29]:


df.isnull().sum()


# In[31]:


print(f"There are {df.isnull().sum().sum()} null values in the dataset now. ")


# In[33]:


df.isnull().sum()


# **Reseting the index of the samples in the dataset**

# In[36]:


df = df.reset_index(drop=True)


# **All the unique cities present in the dataset**

# In[60]:


x=df["City"].unique()
x


# In[68]:


print(f"There are {len(x)} cities present in the dataset.")


# **Minmum and maximum ratings of different cities**

# In[87]:


y = df.groupby("City")["Ratings"].agg(["max", "min"]).reset_index()
print(y)


# **Visualisation**

# In[107]:


plt.figure(figsize=(17, 17))
sns.boxplot(x="City", y="Ratings", data=df, hue="City", palette="Set2", dodge=False, legend=False)
plt.title("Boxplot of Ratings by City")
plt.xlabel("City")
plt.ylabel("Rating")
plt.xticks(rotation=90) 
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


# In[103]:


cities = df["City"].unique()
for city in cities:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="City", y="Ratings", data=df[df["City"] == city], hue="City", palette="Set2", dodge=False)
    plt.title(f"Boxplot of Ratings for {city}")
    plt.xlabel("City")
    plt.ylabel("Ratings")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


# In[52]:


df.head()


# **Recommender function**

# In[79]:


model = NearestNeighbors(n_neighbors=3, metric="euclidean")


# In[81]:


def recommend_trip(city, rating, distance, duration):
    city_data = df[df["City"].str.lower() == city.lower()]

    if city_data.empty:
        print(f"\nSorry, we don't have recommendations for the city '{city}'.")
        return


    filtered_data = city_data[city_data["Ratings"] >= rating]

    if filtered_data.empty:
        print(f"\nNo recommendations in {city} with a rating of {rating} or higher.")
        return

    features = filtered_data[["Ratings", "Distance(Km)", "Time Duration"]].values
    model.fit(features)

    query = [[rating, distance, duration]]

    distances, indices = model.kneighbors(query)


    recommendations = filtered_data.iloc[indices[0]]

    print(f"\n\nRecommended Places and Activities for Your Trip in {city} 🚗:")
    for _, row in recommendations.iterrows():
        print(f"- {row['Place']} (Rating: {row['Ratings']})")
        display(HTML(f"<img src='{row['Images']}' style='width:300px;height:200px;'>"))


# **Taking user inputs such as city name, minimum ratings, maximum distance in km, maximum time the user has for exploration**

# In[84]:


user_city = input("Enter the city: ").strip()
user_rating = float(input("Enter your preferred minimum rating (e.g., 4.0): "))
user_distance = float(input("Enter your maximum distance from the city center (in km): "))
user_duration = float(input("Enter the maximum time you have for exploring (in hours): "))

recommend_trip(user_city, user_rating, user_distance, user_duration)


# **Above are the recommended places and activities for the requested city**

# **Saving the model and the cleaned dataset for deployment using streamlit**

# In[86]:


import joblib

df.to_csv("cleaned_trip_data.csv", index=False)  # Saving cleaned dataset
joblib.dump(model, "trip_recommendation_model.pkl")  # Saving the model


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




