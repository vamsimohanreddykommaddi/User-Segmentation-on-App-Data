# User Segmentation on App Data using Clustering

## 📌 Project Overview
In this project, I performed **User Segmentation** using clustering techniques to analyze user behavior in a mobile app. The goal was to group users based on engagement patterns and identify segments such as **Retained Users, Churned Users, and At-Risk Users**. This insight can help businesses improve user retention strategies.

## 📂 Dataset
The dataset contains user interaction details such as:
- `userid` - Unique identifier for each user
- `Average Screen Time` - Average time spent on the app per session
- `Average Spent on App (INR)` - Amount spent by the user on the app
- `Left Review` - Whether the user left a review (Yes/No)
- `Ratings` - User's rating of the app
- `New Password Requests` - Number of password reset requests
- `Last Visited Minutes` - Minutes spent in the last session
- `Status` - Whether the user **Installed** or **Uninstalled** the app

## 🔍 Exploratory Data Analysis (EDA)
- Performed **AutoEDA** using `sweetviz`
- Checked for missing values and duplicates
- Visualized relationships between engagement metrics and user retention

## ⚙️ Data Preprocessing
- **Outlier Handling**: Used `Winsorization` to cap extreme values
- **Feature Scaling**: Applied `MinMaxScaler` to normalize numerical features

## 🏆 Clustering Techniques Applied
1. **Hierarchical Clustering (Agglomerative)**
   - Plotted **Dendrogram** to determine the optimal number of clusters
   - Assigned users to **three distinct clusters**
   - Evaluated using **Silhouette Score**

2. **K-Means Clustering**
   - Used **Elbow Method** to determine optimal clusters
   - Assigned users to **three clusters**: `Retained`, `Churn`, `Need Attention`
   - Visualized segments with **scatter plots**

## 📊 Results & Insights
- Users with **low screen time and spending** tend to churn
- Users who **rated the app lower** also had lower engagement
- Segmentation provided actionable insights for **retention strategies**

## 🛠 Technologies Used
- **Python** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
- **Clustering Algorithms** (K-Means, Hierarchical)
- **EDA & Visualization** (Sweetviz, Boxplots, Scatterplots)

## 🚀 How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/vamsimohanreddykommaddi/User-Segmentation-on-App-Data.git
   cd user-segmentation
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python user_segmentation.py
   ```

## 📌 Key Takeaways
✅ Clustering helps in **user behavior analysis**
✅ Insights can guide **marketing and retention strategies**
✅ Data-driven decisions improve **user engagement**

📢 **If you find this project helpful, don't forget to ⭐ the repo!** 🚀

---

📧 Contact: vamsimohan2122@gmail.com

