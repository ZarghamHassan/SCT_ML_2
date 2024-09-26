# %%
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# %%
df=pd.read_csv('Mall_Customers.csv')
df.head

# %%
print(df.isnull().sum())

# %%
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler=StandardScaler()
features_scaled = scaler.fit_transform(features)

# %%
wcss= []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow graph
plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Clusters')
plt. xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# %%
kmeans = KMeans(n_clusters=5, init='k-means++',random_state=42)
df['Cluster']=kmeans.fit_predict(features_scaled)


# %%
plt.figure(figsize=(10,7))
sns.scatterplot(x=features_scaled[:,1], y=features_scaled[:, 2], hue=df['Cluster'], palette='Set1' )
plt.title('Cluster Of Customers')
plt.xlabel('Annual Incomed (Scaled)')
plt.ylabel('Spending Score (Scaled)')
plt.show()

# %%
# Perform PCA for 2D visualization
pca = PCA(n_components=2)
pca_features = pca.fit_transform(features_scaled)

# Plot the clusters in 2D
plt.figure(figsize=(10,7))
sns.scatterplot(x=pca_features[:,0], y=pca_features[:,1], hue=df['Cluster'], palette='Set1')
plt.title('2D PCA Cluster Visualization')
plt.show()


# %%
# Plot the distributions of different features across clusters
sns.boxplot(x='Cluster', y='Annual Income (k$)', data=df)
plt.title('Annual Income Distribution per Cluster')
plt.show()

sns.boxplot(x='Cluster', y='Spending Score (1-100)', data=df)
plt.title('Spending Score Distribution per Cluster')
plt.show()


# %%
score = silhouette_score(features_scaled,df['Cluster'])
print("Silhouette Score: ", score)

# %%
import streamlit as st
import pandas as pd

# Assuming df is already defined as your DataFrame
st.title("Customer Segmentation Dashboard")

# Display data table
st.write("Data Overview:")
st.write(df.head())

# Show summary statistics
st.write("Summary Statistics by Cluster:")
st.write(df.groupby('Cluster').agg({'Age': 'mean', 'Annual Income (k$)': 'mean', 'Spending Score (1-100)': 'mean'}))

# Cluster size bar chart
st.write("Cluster Size Distribution:")
st.bar_chart(df.groupby('Cluster')['CustomerID'].count())

# Add a filter for gender
gender_filter = st.selectbox("Select Gender", options=["All", "Male", "Female"])
if gender_filter != "All":
    df = df[df['Gender'] == gender_filter]

# Provide a download button for the filtered data
csv = df.to_csv(index=False)
st.download_button("Download Filtered Data", csv, "filtered_data.csv", "text/csv")


