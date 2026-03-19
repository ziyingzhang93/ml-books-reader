import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

# Load the dataset
Ames = pd.read_csv("Ames.csv")

# Convert the below numeric features to categorical features
Ames["MSSubClass"] = Ames["MSSubClass"].astype("object")
Ames["YrSold"] = Ames["YrSold"].astype("object")
Ames["MoSold"] = Ames["MoSold"].astype("object")

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

ordinal_features = list(ordinal_order.keys())
nominal_features = [feat for feat in categorical_features if feat not in ordinal_features]

print(len(numeric_features) +
      len(ordinal_features) +
      Ames[nominal_features].infer_objects(copy=False).fillna("None").nunique().sum())
