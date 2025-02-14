import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

df = pd.read_csv('/content/youtube.csv')

# Mapping floating-point values back to category names
float_category_mapping = {
    1: 'Trailers',
    0.571428571: 'News',
    0.547619048: 'Updates',
    0.523809524: 'Entertainment',
    0.5: 'Comedy Vlogs',
    0.214285714: 'Music',
    0.452380952: 'Other Category',  # Adjust if needed
    1: 'Series',
    0.642857143: 'Tech Videos',
    0.619047619: 'Concert Videos',
    0.595238095: 'Telugu News',
    0.428571429: 'Telugu News'  # Duplicate category, consider renaming if appropriate
}

# Apply mapping to the dataset (assuming you have a column that corresponds to the floating-point mapping)
df['category_name'] = df['category_id'].map(float_category_mapping)

# Check for unmapped categories
unmapped_categories = df[df['category_name'].isna()]['category_id'].unique()
if len(unmapped_categories) > 0:
    print(f"Unmapped category IDs: {unmapped_categories}")
else:
    print("All categories are mapped.")

# Define a function to calculate the rating
def calculate_rating(row):
    if row['likes'] + row['dislikes'] == 0:
        return 0  # Avoid division by zero
    return (row['likes'] / (row['likes'] + row['dislikes'])) * 5

# Apply the function to calculate ratings for all videos
df['rating'] = df.apply(calculate_rating, axis=1)

# Determine if a video is "trending" based on rating (e.g., rating > 3.5 is considered trending)
df['trending'] = df['rating'] > 3.5
df['trending'] = df['trending'].astype(int)  # Convert trending to integer (1 for trending, 0 for non-trending)

# Split the dataset into 80% training and 20% testing (9000 samples for testing)
train_data, test_data = train_test_split(df, test_size=0.2, stratify=df['trending'], random_state=42)

# Output the percentage of training and testing data
train_percentage = (len(train_data) / len(df)) * 100
test_percentage = (len(test_data) / len(df)) * 100

print(f"Training Data: {len(train_data)} samples ({train_percentage:.2f}% of total)")
print(f"Testing Data: {len(test_data)} samples ({test_percentage:.2f}% of total)")

# Count the number of trending and non-trending videos in training and testing sets
train_trending_count = train_data['trending'].value_counts()
test_trending_count = test_data['trending'].value_counts()

print(f"\nTraining Data - Trending Videos Count:\n{train_trending_count}")
print(f"\nTesting Data - Trending Videos Count:\n{test_trending_count}")

# Features and labels for XGBoost
X_train = train_data[['views', 'likes', 'dislikes', 'comment_count']]  # Include features for the model
y_train = train_data['trending']
X_test = test_data[['views', 'likes', 'dislikes', 'comment_count']]
y_test = test_data['trending']

# Train the XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Output the evaluation metrics
print(f"\nAccuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)
print(f"ROC AUC Score: {roc_auc:.4f}")

# 1. Bar chart of average ratings per category
avg_ratings_per_category = df.groupby('category_name')['rating'].mean()

plt.figure(figsize=(5, 4))
avg_ratings_per_category.plot(kind='bar', color='skyblue')
plt.title('Average Ratings by Category')
plt.xlabel('Category')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Bar chart of number of videos per category
videos_per_category = df['category_name'].value_counts()

plt.figure(figsize=(5, 4))
videos_per_category.plot(kind='bar', color='lightgreen')
plt.title('Number of Videos per Category')
plt.xlabel('Category')
plt.ylabel('Number of Videos')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Bar chart of liked (trending) videos per category
liked_videos_per_category = df[df['trending'] == 1]['category_name'].value_counts()

plt.figure(figsize=(5, 4))
liked_videos_per_category.plot(kind='bar', color='coral')
plt.title('Number of Liked (Trending) Videos per Category')
plt.xlabel('Category')
plt.ylabel('Number of Liked Videos')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Pie chart of overall trending vs non-trending videos
total_trending = liked_videos_per_category.sum()
total_non_trending = len(df) - total_trending

labels = ['Trending', 'Non-Trending']
sizes = [total_trending, total_non_trending]
colors = ['gold', 'lightcoral']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Overall Trending vs Non-Trending Videos')
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
plt.show()

# 5. Positive vs Negative Comments Count
positive_comments_count = df['comment_count'][df['trending'] == 1].sum()
negative_comments_count = df['comment_count'][df['trending'] == 0].sum()

# Pie chart for positive vs negative comments
labels = ['Positive Comments', 'Negative Comments']
sizes = [positive_comments_count, negative_comments_count]
colors = ['lightblue', 'salmon']

plt.figure(figsize=(5,5))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Positive vs Negative Comments')
plt.axis('equal')
plt.show()
