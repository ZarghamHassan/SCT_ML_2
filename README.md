# Customer Segmentation Using KMeans Clustering
## Overview
This project implements KMeans clustering to perform customer segmentation based on demographic and behavioral data from a mall customers dataset. The primary goal is to identify distinct customer groups that can be targeted for personalized marketing strategies.
## Dataset
The dataset used in this project is the `Mall_Customers.csv`, which contains information about mall customers, including:
- **CustomerID**: Unique identifier for each customer
- **Gender**: Gender of the customer
- **Age**: Age of the customer
- **Annual Income (k$)**: Annual income of the customer in thousands of dollars
- **Spending Score (1-100)**: A score assigned to each customer based on their spending behavior

  ## Key Features

- **Data Preprocessing**: Includes data cleaning and scaling of features using `StandardScaler` to prepare the data for clustering.
- **Elbow Method**: Generates an elbow graph to help determine the optimal number of clusters for KMeans.
- **Clustering**: Applies KMeans clustering to segment customers into distinct groups based on their age, annual income, and spending score.
- **Visualization**:
  - Scatter plots to visualize the customer clusters in the scaled feature space and in 2D PCA space.
  - Box plots to show the distribution of annual income and spending score across different clusters.
- **Silhouette Score**: Calculates the silhouette score to evaluate the quality of the clusters formed.
- **Streamlit Dashboard**: An interactive dashboard that displays the results of the clustering analysis, allowing users to visualize customer segments and explore the underlying data in real time.

 ## Technologies Used
 + **Pandas**
 + **Scikit-Learn**
 + **Matplotlib**
 + **Seaborn**
 + Streamlit
## Installation
To run this project, ensure you have the following packages installed:
```bash
pip install pandas scikit-learn matplotlib seaborn streamlit
```

## Usage

1. **Explore the Jupyter Notebook**:
   - Open `customer_segmentation_analysis.ipynb` to review the data analysis and clustering steps performed in this project. This notebook provides detailed insights into the customer segmentation process.

2. **Run the Streamlit app**:
   - Use the Python script named `customer_segmentation_app.py` for an interactive dashboard.
   - In your terminal, navigate to the project directory and run the command:
     ```bash
     streamlit run customer_segmentation_app.py
     ```

3. **Access the dashboard**:
   - Open your web browser and go to `http://localhost:8501` to view the customer segmentation dashboard.
