'''
Problem statement : 
    In the problem of app user segmentation, we need to group users based on how they engage with the app.
'''
'''
I found a dataset containing data about how users who use an app daily and the users who have uninstalled the app 
engaged with the app. This data can be used to group users to find the retained and churn users.
'''

# importing necessary libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz


from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn import metrics


# Loading the data into a pandas dataframe

user_data = pd.read_csv(r"D:\Infotact-projects\userbehaviour-appdata.csv")

'''Data Dictionary: Description about each feature in the dataset

        userid - unique identifier of each user
        Average Screen time - Average Screen usage of user
        Average Spent on App(INR) - Average money spent on the app
        Left Review - Boolean feature which indicates whether user gave the review about the app or not
        Ratings - Ratings given by users for the app
        New Password Requests - Number of new password requests user made while using the app (forgot password)
        Last Visited Minutes - Number of minutes user spent on the app on the last visit
        Status - categoric feature which indicates whether user installed or uninstalled the app.
        
'''

# Displaying the information of dataset

user_data.info()


# Exploratory Data Analysis (EDA)

# Generating descriptive statistics for dataset

user_data.describe()

# Performing AutoEDA on the dataset using sweetviz library

user_report = sweetviz.analyze(user_data)

user_report.show_html('Report.html')


# Data Preprocessing

# checking for missing values in the dataset

user_data.isnull().sum() # There are no missing values in the dataset

# checking for duplicates in the dataset

user_data.duplicated().sum() # there are no duplicates in the dataset

# Checking for outliers in the numerical columns using boxplot

sns.boxplot(user_data['Average Screen Time'])
plt.show()

sns.boxplot(user_data['Average Spent on App (INR)'])
plt.show()

sns.boxplot(user_data['Ratings'])
plt.show()

sns.boxplot(user_data['Last Visited Minutes'])
plt.show()

'''from boxplots we understood that there are outliers in the column 'Last Visited Minutes' '''

# Handling outliers using winsorization technique

# creating a winsorizer object

winsor_iqr = Winsorizer(capping_method='iqr', tail = 'both', fold = 1.5, variables = 'Last Visited Minutes')

# Using winsorizer object to handle the outliers

user_data = winsor_iqr.fit_transform(user_data)

'''Now we will plot the relationship between the spending capacity and screen time of the active users 
and the users who have uninstalled the app'''

plt.figure(figsize = (10,8))
scatter = sns.scatterplot(data = user_data, 
                          x = "Average Screen Time", 
                          y = "Average Spent on App (INR)", 
                          size = "Average Spent on App (INR)", 
                          hue = "Status",
                          palette = 'deep',
                          sizes = (30,300))
plt.show()

'''Users who uninstalled the app had an average screen time of fewer than 5 minutes a day, 
and the average spent was less than 100. We can also see a linear relationship between the average screen time 
and the average spending of the users still using the app.'''

'''Now we will plot the the relationship between the ratings given by users and the average screen time'''

plt.figure(figsize = (10,8))
scatter2 = sns.scatterplot(data = user_data, 
                          x = "Average Screen Time", 
                          y = "Ratings", 
                          size = "Ratings", 
                          hue = "Status",
                          palette = 'deep',
                          sizes = (30,300))
plt.show()

'''we can see that users who uninstalled the app gave the app a maximum of five ratings. 
Their screen time is very low compared to users who rated more. So, this describes that users who don’t like 
to spend more time rate the app low and uninstall it at some point.'''

# Now we will consider the features for clustering

clustering_data = pd.DataFrame(user_data[["Average Screen Time","Last Visited Minutes", 
                        "Average Spent on App (INR)","Ratings",]])

# Now we will scale these features so that to make it ready for model building phase

scaler = MinMaxScaler()

clustering_data_scaled = scaler.fit_transform(clustering_data)

clustering_data_scaled = pd.DataFrame(clustering_data_scaled, columns = clustering_data.columns)


# Now displaying the cleaned data

clustering_data_scaled.head()


''' Now we will build clustering models using the cleaned data'''

'''Performing hierarchical clustering - Agglomerative clustering'''   

#plotting a dendrogram

plt.figure(1, figsize=(16, 8))  # Creating a new figure with specified size for the dendrogram plot

# Generating a dendrogram plot using hierarchical clustering with complete linkage method
tree_plot = dendrogram(linkage(clustering_data_scaled, method="complete"))

plt.title('Hierarchical Clustering Dendrogram')  # Setting the title of the dendrogram plot
plt.xlabel('Index')  # Setting the label for x-axis
plt.ylabel('Euclidean distance')  # Setting the label for y-axis
plt.show()  # Displaying the dendrogram plot

# Applying AgglomerativeClustering and grouping data into 3 clusters
# based on the above dendrogram as a reference
# Creating an instance of AgglomerativeClustering with parameters: 
# - n_clusters: number of clusters set to 3
# - metric(affinity): distance metric set to 'euclidean'
# - linkage: linkage criterion set to 'ward'

hc_model = AgglomerativeClustering(n_clusters = 3, metric = 'euclidean', linkage = 'complete')

# fitting agglomerative clustering model to data and predicting labels for each sample

y_hcm = hc_model.fit_predict(clustering_data_scaled)

# converting cluster labels into pandas series for further analysis

cluster_labels_hc = pd.Series(hc_model.labels_)

'''Cluster evaluation using silhouette score'''

# silhouette score for data with 3 clusters

metrics.silhouette_score(clustering_data_scaled, cluster_labels_hc)

# Plotting the clusters obtained from hierarchical clustering
plt.figure(1)
plt.title("user segments from Hierarchical clustering")
plt.xlabel('Last Visited Minutes')
plt.ylabel('Average Spent on App (INR)')
plt.scatter(x = user_data['Last Visited Minutes'], y = user_data['Average Spent on App (INR)'], c=cluster_labels_hc, s=50, cmap='tab20b')
plt.show()


'''Performing K-means clustering'''

#scree plot or elbow curve
TWSS = []  # List to store the total within-cluster sum of squares (TWSS) for each value of k
k = list(range(2, 9))  # List of values of k (number of clusters) to be evaluated

# Looping through each value of k
for i in k:
    kmeans = KMeans(n_clusters=i)  # Creating a KMeans clustering model with i clusters
    kmeans.fit(clustering_data_scaled)  # Fitting the KMeans model to the cleaned numeric data
    TWSS.append(kmeans.inertia_)  # Appending the total within-cluster sum of squares (TWSS) to the list TWSS

# Displaying the TWSS values for each value of k
TWSS

# Creating a scree plot to visualize the relationship between the number of clusters and TWSS
plt.plot(k, TWSS, 'ro-')  # Plotting the values of k (x-axis) against the TWSS (y-axis)
plt.xlabel("No_of_Clusters")  # Labeling the x-axis as "No_of_Clusters"
plt.ylabel("total_within_SS")  # Labeling the y-axis as "total_within_SS"
plt.show()

# Using KneeLocator
List = []

for k in range(2, 9):
    kmeans = KMeans(n_clusters = k, init = "random", max_iter = 30, n_init = 10) 
    kmeans.fit(clustering_data_scaled)
    List.append(kmeans.inertia_)

from kneed import KneeLocator
#kl = KneeLocator(range(2, 9), List, curve = 'convex')
kl = KneeLocator(range(2, 9), List, curve='convex', direction = 'decreasing')
kl.elbow
plt.style.use("ggplot")
plt.plot(range(2, 9), List)
plt.xticks(range(2, 9))
plt.ylabel("Interia")
plt.axvline(x = kl.elbow, color = 'r', label = 'axvline - full height', ls = '--')
plt.show()

# Creating a KMeans clustering model with 3 clusters
kmeans_model = KMeans(n_clusters = 3)

# Fitting the KMeans model to the cleaned numeric data

y_kmeans = kmeans_model.fit_predict(clustering_data_scaled)

# Obtaining the cluster labels assigned by the KMeans model to each data point

cluster_labels_kmeans = pd.Series(kmeans_model.labels_)

# Cluster Evaluation for Kmeans clusters

kmeans_score = metrics.silhouette_score(clustering_data_scaled, cluster_labels_kmeans)

kmeans_score

# Plotting the clusters obtained from KMeans
plt.figure(2)
plt.title("user segments from K-Means")
plt.xlabel('Last Visited Minutes')
plt.ylabel('Average Spent on App (INR)')
plt.scatter(x = user_data['Last Visited Minutes'], y = user_data['Average Spent on App (INR)'], c=cluster_labels_kmeans, s=50, cmap='tab20b')
plt.show()

# adding cluster segments to original data

user_data['Segments'] = cluster_labels_kmeans


user_data['Segments'].value_counts()

user_data["Segments"] = user_data["Segments"].map({0: "Retained", 1: 
    "Churn", 2: "Need Attention"})


# finding Average Screen time and Average Spent on App based on segments

# Calculating the average screen time per cluster
avg_screen_time_per_segment = user_data.groupby("Segments")["Average Screen Time"].mean()
avg_screen_time_per_segment

# Calculating the average spent on App per based on segments
avg_spent_on_app_per_segment = user_data.groupby("Segments")["Average Spent on App (INR)"].mean()
avg_spent_on_app_per_segment

# Calculating the Status based on segments
status_distribution = user_data.groupby("Segments")["Status"].value_counts().unstack()

status_distribution

# visualizing the segments

# Define colors for each segment
colors = {'Retained': 'green', 'Need Attention': 'blue', 'Churn': 'red'}

plt.figure(figsize=(10, 8))

# Loop through unique segments and plot
for segment in user_data["Segments"].unique():
    subset = user_data[user_data["Segments"] == segment]
    sns.scatterplot(x=subset['Last Visited Minutes'], 
                    y=subset['Average Spent on App (INR)'], 
                    color=colors.get(segment, 'red'), 
                    label=segment, 
                    edgecolor='black', 
                    s=50)

# Labels and title
plt.xlabel('Last Visited Minutes', fontsize=12)
plt.ylabel('Average Spent on App (INR)', fontsize=12)
plt.title('Customer Segments Analysis', fontsize=14)
plt.legend(title='Segments')
plt.grid(True)

plt.show()

''' 
--> The green segment shows the segment of users the app has retained over time. 
--> The blue segment indicates the segment of users who just uninstalled the app or are about 
    to uninstall it soon. 
--> And the red segment indicates the segment of users that the application has lost.

'''







