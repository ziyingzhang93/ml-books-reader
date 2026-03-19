import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Load dataset
data = pd.read_csv("Ames.csv")
X = data.drop(["SalePrice"], axis=1)
y = data["SalePrice"]
cat_features = [col for col in X.columns if X[col].dtype == "object"]
X["Electrical"] = X["Electrical"].fillna(X["Electrical"].mode()[0])
X[cat_features] = X[cat_features].fillna("Missing")
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# Set up k-fold cross-validation
kf = KFold(n_splits=5)
feature_importances = []

# Iterate over each split
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train default CatBoost model
    model = CatBoostRegressor(cat_features=cat_features, random_state=42, verbose=0)
    model.fit(X_train, y_train)
    feature_importances.append(model.get_feature_importance())

# Average feature importance across all folds
avg_importance = np.mean(feature_importances, axis=0)

# Convert to DataFrame
feat_imp_df = pd.DataFrame({"Feature": X.columns, "Importance": avg_importance})

# Sort and take the top 20 features
top_features = feat_imp_df.sort_values(by="Importance", ascending=False).head(20)

# Set the style and color palette
sns.set_style("whitegrid")
palette = sns.color_palette("rocket", len(top_features))

# Create the plot
plt.figure(figsize=(12, 10))
ax = sns.barplot(x="Importance", y="Feature", hue="Feature",
                 data=top_features, palette=palette, legend=False)

# Customize the plot
plt.title("Top 20 Most Important Features - CatBoost Model",
          fontsize=20, fontweight="bold")
plt.xlabel("Importance Score", fontsize=15)
plt.ylabel("Features", fontsize=15)

# Add value labels to the end of each bar
for i, v in enumerate(top_features["Importance"]):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=13)

# Extend x-axis by 10% and feature names font size
plt.xlim(0, max(top_features["Importance"]) * 1.1)
plt.yticks(fontsize=13)

# Adjust layout and display
plt.tight_layout()
plt.show()
