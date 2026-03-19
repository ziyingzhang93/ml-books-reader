import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

# Load the dataset
Ames = pd.read_csv("Ames.csv")

# Convert the below numeric features to categorical features
Ames["MSSubClass"] = Ames["MSSubClass"].astype("object")
Ames["YrSold"] = Ames["YrSold"].astype("object")
Ames["MoSold"] = Ames["MoSold"].astype("object")

# Identify numeric and categorical features from the dataset based on the data type
numeric_features = Ames.select_dtypes(include=["int64", "float64"]) \
                       .drop(columns=["PID", "SalePrice"]).columns
categorical_features = Ames.select_dtypes(include=["object"]).columns \
                           .difference(["Electrical"])

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

# Helper function to fill "None" for missing categorical data
def fill_none(X):
    return X.infer_objects(copy=False).fillna("None")

# Pipeline for "Electrical": Fill missing value with mode then apply ordinal encoding
electrical_transformer = Pipeline(steps=[
    ("impute_electrical", SimpleImputer(strategy="most_frequent")),
    ("ordinal_electrical", OrdinalEncoder(categories=[ordinal_order["Electrical"]]))
])

# Pipeline for numeric features: Impute missing values using mean
numeric_transformer = Pipeline(steps=[
    ("impute_mean", SimpleImputer(strategy="mean"))
])

# Pipeline for ordinal features: Fill missing values with "None" then apply
# ordinal encoding
ordinal_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
    ("ordinal", OrdinalEncoder(categories=[ordinal_order[feat]
                                           for feat in ordinal_features
                                           if feat in ordinal_except_electrical]))
])

# Pipeline for nominal categorical features: Fill missing values with "None" then apply
# one-hot encoding
nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]
categorical_transformer = Pipeline(steps=[
    ("fill_none", FunctionTransformer(fill_none, validate=False)),
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

# Define the full model pipeline
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", DecisionTreeRegressor(random_state=42))
])

# Evaluate the model using cross-validation
scores = cross_val_score(model_pipeline, Ames.drop(columns="SalePrice"), Ames["SalePrice"])

# Output the result
print("Decision Tree Regressor Mean CV R^2:", round(scores.mean(),4))
