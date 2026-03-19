import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

Ames = pd.read_csv("Ames.csv")

# Adjust data types for categorical variables
for col in ["MSSubClass", "YrSold", "MoSold"]:
    Ames[col] = Ames[col].astype("object")

# Exclude "PID" and "SalePrice" from features and handle the "Electrical" column
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])
electrical_feature = ["Electrical"]

# Manually specify the categories for ordinal encoding according to the data dictionary
ordinal_order = {
    # Electrical system
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    # General shape of property
    "LotShape": ["IR3", "IR2", "IR1", "Reg"],
    # Type of utilities available
    "Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
    # Slope of property
    "LandSlope": ["Sev", "Mod", "Gtl"],
    # Evaluates the quality of the material on the exterior
    "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Evaluates the present condition of the material on the exterior
    "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Height of the basement
    "BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # General condition of the basement
    "BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Walkout or garden level basement walls
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    # Quality of basement finished area
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    # Quality of second basement finished area
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    # Heating quality and condition
    "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Kitchen quality
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    # Home functionality
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    # Fireplace quality
    "FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Interior finish of the garage
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
    # Garage quality
    "GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Garage condition
    "GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    # Paved driveway
    "PavedDrive": ["N", "P", "Y"],
    # Pool quality
    "PoolQC": ["None", "Fa", "TA", "Gd", "Ex"],
    # Fence quality
    "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"]
}

# Extract list of ALL ordinal features from dictionary
ordinal_features = list(ordinal_order.keys())

# List of ordinal features except Electrical
ordinal_except_electrical = [feat for feat in ordinal_features if feat != "Electrical"]

# Define transformations for various feature types
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])

numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])

# Updated categorical imputer using SimpleImputer
categorical_imputer = SimpleImputer(strategy="constant", fill_value="None")

ordinal_transformer = Pipeline([
    ("impute_ordinal", categorical_imputer),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])

nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline([
    ("impute_nominal", categorical_imputer),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combined preprocessor for numeric, ordinal, nominal, and specific electrical data
preprocessor = ColumnTransformer(
    transformers=[
        ("electrical", electrical_transformer, ["Electrical"]),
        ("num", numeric_transformer, numeric_features),
        ("ordinal", ordinal_transformer, ordinal_except_electrical),
        ("nominal", categorical_transformer, nominal_features)
])

# "preprocessor" is already set up as your preprocessing pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(random_state=42))
])

# Parameter grid for GridSearchCV
param_grid = {
    "regressor__learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
    "regressor__n_estimators": [100, 200, 300, 400, 500]
}

# Setup the GridSearchCV
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="r2", verbose=1)

# Fit the GridSearchCV to the data
grid_search.fit(Ames.drop(columns="SalePrice"), Ames["SalePrice"])

# Best parameters and best score from Grid Search
print("Best parameters (Grid Search):", grid_search.best_params_)
print("Best score (Grid Search):", round((grid_search.best_score_), 4))
