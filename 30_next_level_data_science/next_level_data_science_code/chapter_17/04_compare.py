# Importing libraries to compare feature importance between GBDT and GOSS:
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data
data = pd.read_csv("Ames.csv")
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]
categorical_cols = X.select_dtypes(include=["object"]).columns
X[categorical_cols] = X[categorical_cols].apply(lambda x: x.astype("category"))

# Set up K-fold cross-validation
kf = KFold(n_splits=5)
gbdt_feature_importances = []
goss_feature_importances = []

# Iterate over each split
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train GBDT model with optimal num_leaves
    gbdt_model = lgb.LGBMRegressor(boosting_type="gbdt", num_leaves=10, verbose=-1)
    gbdt_model.fit(X_train, y_train)
    gbdt_feature_importances.append(gbdt_model.feature_importances_)

    # Train GOSS model with optimal num_leaves
    goss_model = lgb.LGBMRegressor(boosting_type="goss", num_leaves=10, verbose=-1)
    goss_model.fit(X_train, y_train)
    goss_feature_importances.append(goss_model.feature_importances_)

# Average feature importance across all folds for each model
avg_gbdt_feature_importance = np.mean(gbdt_feature_importances, axis=0)
avg_goss_feature_importance = np.mean(goss_feature_importances, axis=0)

# Convert to DataFrame
feat_importances_gbdt = pd.DataFrame({"Feature": X.columns,
                                      "Importance": avg_gbdt_feature_importance})
feat_importances_goss = pd.DataFrame({"Feature": X.columns,
                                      "Importance": avg_goss_feature_importance})

# Sort and take the top 10 features
top_gbdt_features = feat_importances_gbdt \
                    .sort_values(by="Importance", ascending=False) \
                    .head(10)
top_goss_features = feat_importances_goss \
                    .sort_values(by="Importance", ascending=False) \
                    .head(10)

# Plotting
plt.figure(figsize=(16, 12))
plt.subplot(1, 2, 1)
sns.barplot(data=top_gbdt_features, y="Feature", x="Importance",
            hue="Feature", orient="h", legend=False, palette="viridis")
plt.title("Top 10 LightGBM GBDT Features", fontsize=18)
plt.xlabel("Importance", fontsize=16)
plt.ylabel("Feature", fontsize=16)
plt.xticks(fontsize=13)
plt.yticks(fontsize=14)

plt.subplot(1, 2, 2)
sns.barplot(data=top_goss_features, y="Feature", x="Importance",
            hue="Feature", orient="h", legend=False, palette="viridis")
plt.title("Top 10 LightGBM GOSS Features", fontsize=18)
plt.xlabel("Importance", fontsize=16)
plt.ylabel("Feature", fontsize=16)
plt.xticks(fontsize=13)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()
