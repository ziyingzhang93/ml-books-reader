# Import necessary libraries for preprocessing and modeling
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import \
    GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor

# Load the dataset
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

# Define model pipelines including Gradient Boosting Regressor
models = {
    "Decision Tree (1 Tree)": DecisionTreeRegressor(random_state=42),
    "Bagging Regressor (100 Decision Trees)":
        BaggingRegressor(estimator=DecisionTreeRegressor(random_state=42),
                         n_estimators=100, random_state=42),
    "Bagging Regressor (200 Decision Trees)":
        BaggingRegressor(estimator=DecisionTreeRegressor(random_state=42),
                         n_estimators=200, random_state=42),
    "Random Forest (Default of 100 Trees)":
        RandomForestRegressor(random_state=42),
    "Random Forest (200 Trees)":
        RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting Regressor (Default of 100 Trees)":
        GradientBoostingRegressor(random_state=42),
    "Gradient Boosting Regressor (200 Trees)":
        GradientBoostingRegressor(n_estimators=200, random_state=42)
}

# Evaluate models using cross-validation and print results
results = {}
for name, model in models.items():
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    scores = cross_val_score(model_pipeline,
                             Ames.drop(columns="SalePrice"), Ames["SalePrice"], cv=5)
    results[name] = round(scores.mean(), 4)
    print(f"{name}: Mean CV R^2 = {results[name]}")
