# ============================================================
# TELCO CUSTOMER CHURN PREDICTION PROJECT
# ============================================================
#
# BUSINESS PROBLEM
# ------------------------------------------------------------
# A telecom company wants to predict which customers are likely
# to churn (leave the company). Before building the machine
# learning model, exploratory data analysis and feature
# engineering steps are required.
#
# DATASET STORY
# ------------------------------------------------------------
# The Telco Customer Churn dataset contains information about
# 7,043 customers of a fictional telecom company in California.
# The company provides home phone and internet services.
# The dataset includes which customers stayed, left, or signed up.
#
# OBSERVATIONS AND VARIABLES
# ------------------------------------------------------------
# 7043 observations, 21 variables
#
# VARIABLES
# ------------------------------------------------------------
# customerID       : Customer ID
# gender           : Gender
# SeniorCitizen    : Whether the customer is a senior citizen (1, 0)
# Partner          : Whether the customer has a partner (Yes, No)
# Dependents       : Whether the customer has dependents (Yes, No)
# tenure           : Number of months the customer has stayed
# PhoneService     : Whether the customer has phone service (Yes, No)
# MultipleLines    : Whether the customer has multiple lines
# InternetService  : Customer’s internet service provider
# OnlineSecurity   : Whether the customer has online security
# OnlineBackup     : Whether the customer has online backup
# DeviceProtection : Whether the customer has device protection
# TechSupport      : Whether the customer has tech support
# StreamingTV      : Whether the customer uses streaming TV
# StreamingMovies  : Whether the customer uses streaming movies
# Contract         : Contract term of the customer
# PaperlessBilling : Whether the customer uses paperless billing
# PaymentMethod    : Payment method
# MonthlyCharges   : Monthly amount charged
# TotalCharges     : Total amount charged
# Churn            : Whether the customer churned (Yes, No)
#
# PROJECT STEPS
# ------------------------------------------------------------
# TASK 1: EXPLORATORY DATA ANALYSIS
#   Step 1: Capture numerical and categorical variables
#   Step 2: Make required adjustments (data type fixes)
#   Step 3: Examine distributions of variables
#   Step 4: Analyze target relationships
#   Step 5: Check outliers
#   Step 6: Check missing values
#
# TASK 2: FEATURE ENGINEERING
#   Step 1: Handle missing values and outliers
#   Step 2: Create new variables
#   Step 3: Perform encoding
#   Step 4: Apply scaling
#
# TASK 3: MODELING
#   Step 1: Build baseline models and compare metrics
#   Step 2: Apply hyperparameter optimization
#   Step 3: Build final models and evaluate on test set
#
# EXTRA
#   - Save visualizations into the images folder
#   - Plot feature importance
#
# ============================================================


# ============================================================
# 1. LIBRARIES
# ============================================================
import os
import warnings
warnings.simplefilter(action="ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)

# Optional libraries
# The code will still run if they are not installed.
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# ============================================================
# 2. SETTINGS
# ============================================================
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

RANDOM_STATE = 42
DATA_PATH = "Telco-Customer-Churn.csv"

# Create images folder automatically
os.makedirs("images", exist_ok=True)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe(percentiles=[0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Returns categorical columns, numerical columns and cardinal columns.

    cat_cols      : categorical columns
    num_cols      : numerical columns
    cat_but_car   : categorical but cardinal columns
    """

    # Categorical columns
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]

    # Numerical but categorical
    num_but_cat = [
        col for col in dataframe.columns
        if dataframe[col].nunique() < cat_th and dataframe[col].dtype != "O"
    ]

    # Categorical but cardinal
    cat_but_car = [
        col for col in dataframe.columns
        if dataframe[col].nunique() > car_th and dataframe[col].dtype == "O"
    ]

    # Final categorical columns
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Final numerical columns
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables   : {dataframe.shape[1]}")
    print(f"cat_cols    : {len(cat_cols)}")
    print(f"num_cols    : {len(num_cols)}")
    print(f"cat_but_car : {len(cat_but_car)}")
    print(f"num_but_cat : {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
    }))
    print("##########################################")

    if plot:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=dataframe[col_name], order=dataframe[col_name].value_counts().index)
        plt.xlabel(col_name)
        plt.ylabel("Count")
        plt.title(f"Countplot of {col_name}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"images/{col_name}_countplot.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        plt.figure(figsize=(8, 4))
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.tight_layout()
        plt.savefig(f"images/{numerical_col}_hist.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({
        "TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
        "Count": dataframe[categorical_col].value_counts(),
        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)
    }), end="\n\n\n")


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    return False


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    if len(na_columns) == 0:
        print("No missing values.")
        return [] if na_name else None

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


def label_encoder(dataframe, column):
    labelencoder = LabelEncoder()
    dataframe[column] = labelencoder.fit_transform(dataframe[column])
    return dataframe


def one_hot_encoder(dataframe, columns):
    dataframe = pd.get_dummies(data=dataframe, columns=columns, drop_first=True)
    return dataframe


def plot_importance(model, features, num=20, save_path=None):
    if not hasattr(model, "feature_importances_"):
        print(f"{model.__class__.__name__} does not support feature importance.")
        return

    feature_imp = pd.DataFrame({
        "Value": model.feature_importances_,
        "Feature": features.columns
    }).sort_values(by="Value", ascending=False).head(num)

    plt.figure(figsize=(10, 8))
    sns.barplot(x="Value", y="Feature", data=feature_imp)
    plt.title("Feature Importance")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()


def plot_confusion_matrix_custom(y_true, y_pred, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()


# ============================================================
# 4. LOAD DATA
# ============================================================
print("\n" + "=" * 70)
print("TELCO CUSTOMER CHURN PROJECT")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully.\n")


# ============================================================
# TASK 1: EXPLORATORY DATA ANALYSIS
# ============================================================

# ------------------------------------------------------------
# Step 1: Capture numerical and categorical variables
# ------------------------------------------------------------
print("\n##########################################")
print("TASK 1: EXPLORATORY DATA ANALYSIS")
print("##########################################")

print("\n### Step 1: General overview of the dataset")
check_df(df)

print("\n### Step 1: Capture numerical and categorical variables")
cat_cols, num_cols, cat_but_car = grab_col_names(df)

print("\nCategorical Columns:")
print(cat_cols)

print("\nNumerical Columns:")
print(num_cols)

print("\nCategorical but Cardinal Columns:")
print(cat_but_car)


# ------------------------------------------------------------
# Step 2: Make required adjustments
# Fix data type issue in TotalCharges
# ------------------------------------------------------------
print("\n### Step 2: Fix data type problems")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# ------------------------------------------------------------
# Step 3: Examine distributions of variables
# ------------------------------------------------------------
print("\n### Step 3: Distribution of categorical variables")
for col in cat_cols:
    cat_summary(df, col, plot=False)

print("\n### Step 3: Distribution of numerical variables")
for col in num_cols:
    print(f"******************* {col} *******************")
    num_summary(df, col, plot=False)

print("\n### Step 3: Additional grouped summaries")
print(df.groupby(["Churn", "gender"]).agg({
    "tenure": ["mean", "std"],
    "MonthlyCharges": "mean",
    "TotalCharges": "count"
}))

print(df.groupby(["InternetService", "gender"]).agg({
    "MonthlyCharges": "mean",
    "tenure": "mean"
}))

print(df.groupby("PaymentMethod").agg({
    "MonthlyCharges": ["sum", "count"]
}))

print(df.groupby(["Churn", "Contract"]).agg({
    "tenure": "mean",
    "MonthlyCharges": "mean"
}))

# Core EDA visuals
plt.figure(figsize=(6, 4))
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.tight_layout()
plt.savefig("images/churn_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(8, 4))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges by Churn")
plt.tight_layout()
plt.savefig("images/monthlycharges_by_churn.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(8, 4))
sns.boxplot(x="Churn", y="tenure", data=df)
plt.title("Tenure by Churn")
plt.tight_layout()
plt.savefig("images/tenure_by_churn.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# ------------------------------------------------------------
# Step 4: Analyze target relationships
# ------------------------------------------------------------
print("\n### Step 4: Target analysis")

# Convert target to numeric form
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

for col in num_cols:
    target_summary_with_num(df, "Churn", col)

for col in cat_cols:
    if col != "Churn":
        target_summary_with_cat(df, "Churn", col)


# ------------------------------------------------------------
# Step 5: Outlier analysis
# ------------------------------------------------------------
print("\n### Step 5: Outlier analysis")
for col in num_cols:
    print(f"{col}: {check_outlier(df, col)}")

for col in num_cols:
    plt.figure(figsize=(7, 3))
    sns.boxplot(data=df, x=col)
    plt.title(f"{col} Outliers")
    plt.tight_layout()
    plt.savefig(f"images/{col}_boxplot.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# ------------------------------------------------------------
# Step 6: Missing value analysis
# ------------------------------------------------------------
print("\n### Step 6: Missing value analysis")
print(df.isna().sum())
print(df.isnull().sum())

na_columns = missing_values_table(df, na_name=True)


# ============================================================
# TASK 2: FEATURE ENGINEERING
# ============================================================
print("\n##########################################")
print("TASK 2: FEATURE ENGINEERING")
print("##########################################")

# ------------------------------------------------------------
# Step 1: Handle missing values and outliers
# ------------------------------------------------------------
print("\n### Step 1: Handle missing values and outliers")

# In TotalCharges, missing values came from blank strings that
# became NaN after numeric conversion.
# Median is safer than mean against extreme values.
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Outlier handling
for col in num_cols:
    replace_with_thresholds(df, col)

print("\nMissing values after filling:")
print(df.isnull().sum())


# ------------------------------------------------------------
# Step 2: Create new variables
# ------------------------------------------------------------
print("\n### Step 2: Creating new features")

# Tenure groups
df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "NEW_TENURE_YEAR"] = "5-6 Year"

# Engaged customers: one-year or two-year contract
df["NEW_Engaged"] = df["Contract"].apply(
    lambda x: 1 if x in ["One year", "Two year"] else 0
)

# Customers with no protection / backup / support at all
# AND is more correct here, because we want all these services absent
df["NEW_noProt"] = df.apply(
    lambda x: 1 if (
        (x["OnlineBackup"] != "Yes") and
        (x["DeviceProtection"] != "Yes") and
        (x["TechSupport"] != "Yes")
    ) else 0,
    axis=1
)

# Young and not engaged customers
df["NEW_Young_Not_Engaged"] = df.apply(
    lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0,
    axis=1
)

# Total number of services used
service_cols = [
    "PhoneService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]
df["NEW_TotalServices"] = (df[service_cols] == "Yes").sum(axis=1)
df["NEW_TotalServices"] = df["NEW_TotalServices"] + (df["InternetService"] != "No").astype(int)

# Any streaming flag
df["NEW_FLAG_ANY_STREAMING"] = df.apply(
    lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0,
    axis=1
)

# Automatic payment
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(
    lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0
)

# Average monthly charge based on total charges
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)

# Increase ratio
df["NEW_Increase"] = df["NEW_AVG_Charges"] / (df["MonthlyCharges"] + 1e-6)

# Average service fee
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df["NEW_TotalServices"] + 1)

# Additional useful risk features
df["NEW_Senior_Monthly"] = df.apply(
    lambda x: 1 if (x["SeniorCitizen"] == 1) and (x["Contract"] == "Month-to-month") else 0,
    axis=1
)

df["NEW_Risky_Payment_Profile"] = df.apply(
    lambda x: 1 if (x["PaperlessBilling"] == "Yes") and (x["PaymentMethod"] == "Electronic check") else 0,
    axis=1
)

print("\nFirst rows after feature engineering:")
print(df.head())
print("\nNew shape:")
print(df.shape)


# ------------------------------------------------------------
# Step 3: Encoding
# ------------------------------------------------------------
print("\n### Step 3: Encoding")

cat_cols, num_cols, cat_but_car = grab_col_names(df)

binary_cols = [
    col for col in df.columns
    if df[col].dtype not in [np.int64, np.float64] and df[col].nunique() == 2
]

for col in binary_cols:
    label_encoder(df, col)

cat_cols = [
    col for col in cat_cols
    if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]
]

df = one_hot_encoder(df, cat_cols)

print("\nData after encoding:")
print(df.head())
print(df.info())


# ------------------------------------------------------------
# Step 4: Scaling
# ------------------------------------------------------------
print("\n### Step 4: Scaling")

numeric_columns = [
    col for col in df.columns
    if df[col].dtype in [np.int64, np.float64] and df[col].nunique() > 10
]

# Remove ratio feature if you want to keep it raw like the instructor did
if "NEW_Increase" in numeric_columns:
    numeric_columns.remove("NEW_Increase")

# Remove target and customerID if present
numeric_columns = [col for col in numeric_columns if col not in ["Churn", "customerID"]]

scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

print("\nScaling completed.")


# ============================================================
# TASK 3: MODELING
# ============================================================
print("\n##########################################")
print("TASK 3: MODELING")
print("##########################################")

# ------------------------------------------------------------
# Prepare X and y
# ------------------------------------------------------------
y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1, errors="ignore")

# Hold-out split for final test evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)


# ------------------------------------------------------------
# Step 1: Build baseline models and compare them
# ------------------------------------------------------------
print("\n### Step 1: Baseline model comparison")

models = [
    ("LR", LogisticRegression(max_iter=500, random_state=RANDOM_STATE)),
    ("KNN", KNeighborsClassifier()),
    ("CART", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ("RF", RandomForestClassifier(random_state=RANDOM_STATE)),
    ("SVM", SVC(probability=True, random_state=RANDOM_STATE))
]

if XGB_AVAILABLE:
    models.append((
        "XGB",
        XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)
    ))

if LGBM_AVAILABLE:
    models.append((
        "LightGBM",
        LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    ))

if CATBOOST_AVAILABLE:
    models.append((
        "CatBoost",
        CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)
    ))

model_results = []

for name, model in models:
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring=["accuracy", "roc_auc", "recall", "precision", "f1"],
        n_jobs=-1
    )

    row = {
        "Model": name,
        "Accuracy": round(cv_results["test_accuracy"].mean(), 4),
        "AUC": round(cv_results["test_roc_auc"].mean(), 4),
        "Recall": round(cv_results["test_recall"].mean(), 4),
        "Precision": round(cv_results["test_precision"].mean(), 4),
        "F1": round(cv_results["test_f1"].mean(), 4)
    }
    model_results.append(row)

    print(f"########## {name} ##########")
    print(f"Accuracy : {row['Accuracy']}")
    print(f"AUC      : {row['AUC']}")
    print(f"Recall   : {row['Recall']}")
    print(f"Precision: {row['Precision']}")
    print(f"F1       : {row['F1']}")
    print()

results_df = pd.DataFrame(model_results).sort_values(by="AUC", ascending=False)
print("Baseline comparison table:")
print(results_df)

plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="Model", y="AUC")
plt.title("Baseline Model AUC Comparison")
plt.tight_layout()
plt.savefig("images/baseline_model_auc_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# ------------------------------------------------------------
# Step 2: Hyperparameter optimization
# ------------------------------------------------------------
print("\n### Step 2: Hyperparameter optimization")

best_models = {}

# Logistic Regression
lr_model = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
lr_params = {"C": [0.01, 0.1, 1, 10]}

lr_grid = GridSearchCV(
    lr_model,
    lr_params,
    cv=3,
    scoring=["accuracy", "roc_auc", "recall", "precision", "f1"],
    refit="roc_auc",
    n_jobs=-1
)

lr_grid.fit(X_train, y_train)
print("LR Best Params:", lr_grid.best_params_)
print("LR Best AUC   :", lr_grid.best_score_)
best_models["LR"] = LogisticRegression(
    **lr_grid.best_params_,
    max_iter=500,
    random_state=RANDOM_STATE
)

# Random Forest
rf_model = RandomForestClassifier(random_state=RANDOM_STATE)
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

rf_grid = GridSearchCV(
    rf_model,
    rf_params,
    cv=3,
    scoring=["accuracy", "roc_auc", "recall", "precision", "f1"],
    refit="roc_auc",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)
print("RF Best Params:", rf_grid.best_params_)
print("RF Best AUC   :", rf_grid.best_score_)
best_models["RF"] = RandomForestClassifier(
    **rf_grid.best_params_,
    random_state=RANDOM_STATE
)

# XGBoost
if XGB_AVAILABLE:
    xgb_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )
    xgb_params = {
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 6],
        "n_estimators": [100, 300]
    }

    xgb_grid = GridSearchCV(
        xgb_model,
        xgb_params,
        cv=3,
        scoring=["accuracy", "roc_auc", "recall", "precision", "f1"],
        refit="roc_auc",
        n_jobs=-1
    )

    xgb_grid.fit(X_train, y_train)
    print("XGB Best Params:", xgb_grid.best_params_)
    print("XGB Best AUC   :", xgb_grid.best_score_)
    best_models["XGB"] = XGBClassifier(
        **xgb_grid.best_params_,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )

# LightGBM
if LGBM_AVAILABLE:
    lgbm_model = LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    lgbm_params = {
        "learning_rate": [0.01, 0.1],
        "n_estimators": [100, 300],
        "num_leaves": [31, 50]
    }

    lgbm_grid = GridSearchCV(
        lgbm_model,
        lgbm_params,
        cv=3,
        scoring=["accuracy", "roc_auc", "recall", "precision", "f1"],
        refit="roc_auc",
        n_jobs=-1
    )

    lgbm_grid.fit(X_train, y_train)
    print("LGBM Best Params:", lgbm_grid.best_params_)
    print("LGBM Best AUC   :", lgbm_grid.best_score_)
    best_models["LightGBM"] = LGBMClassifier(
        **lgbm_grid.best_params_,
        random_state=RANDOM_STATE,
        verbose=-1
    )


# ------------------------------------------------------------
# Step 3: Final model evaluation on hold-out test set
# ------------------------------------------------------------
print("\n### Step 3: Final model evaluation")

final_results = []

for name, model in best_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        y_prob = None
        auc = np.nan

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    final_results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4) if not np.isnan(auc) else np.nan,
        "Recall": round(rec, 4),
        "Precision": round(prec, 4),
        "F1": round(f1, 4)
    })

    print(f"\n{name} Test Results")
    print(f"Accuracy : {acc:.4f}")
    print(f"AUC      : {auc:.4f}" if not np.isnan(auc) else "AUC      : N/A")
    print(f"Recall   : {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1       : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix_custom(
        y_test,
        y_pred,
        title=f"{name} Confusion Matrix",
        save_path=f"images/{name.lower()}_confusion_matrix.png"
    )

final_results_df = pd.DataFrame(final_results).sort_values(by="AUC", ascending=False)
print("\nFinal test performance table:")
print(final_results_df)

best_model_name = final_results_df.iloc[0]["Model"]
print(f"\nBest final model based on AUC: {best_model_name}")


# ============================================================
# ROC CURVE
# ============================================================
best_model = best_models[best_model_name]
best_model.fit(X_train, y_train)

if hasattr(best_model, "predict_proba"):
    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_estimator(best_model, X_test, y_test)
    plt.title(f"ROC Curve - {best_model_name}")
    plt.tight_layout()
    plt.savefig("images/roc_curve_best_model.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================
print("\n### Feature Importance")

# Best model importance if supported
plot_importance(
    best_model,
    X_train,
    num=20,
    save_path="images/best_model_feature_importance.png"
)

# Also plot for tree-based models if they exist
if "RF" in best_models:
    best_models["RF"].fit(X_train, y_train)
    plot_importance(
        best_models["RF"],
        X_train,
        num=20,
        save_path="images/rf_feature_importance.png"
    )

if "XGB" in best_models:
    best_models["XGB"].fit(X_train, y_train)
    plot_importance(
        best_models["XGB"],
        X_train,
        num=20,
        save_path="images/xgb_feature_importance.png"
    )

if "LightGBM" in best_models:
    best_models["LightGBM"].fit(X_train, y_train)
    plot_importance(
        best_models["LightGBM"],
        X_train,
        num=20,
        save_path="images/lightgbm_feature_importance.png"
    )


# ============================================================
# EXTRA VISUALS FOR GITHUB
# ============================================================
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="MonthlyCharges", hue="Churn", kde=True, element="step")
plt.title("Monthly Charges Distribution by Churn")
plt.tight_layout()
plt.savefig("images/monthlycharges_distribution_by_churn.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="tenure", hue="Churn", kde=True, element="step")
plt.title("Tenure Distribution by Churn")
plt.tight_layout()
plt.savefig("images/tenure_distribution_by_churn.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

original_df_for_plot = pd.read_csv(DATA_PATH)
plt.figure(figsize=(8, 4))
sns.countplot(data=original_df_for_plot, x="Contract", hue="Churn")
plt.title("Contract Type by Churn")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("images/contract_type_by_churn.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# ============================================================
# PROJECT COMPLETION MESSAGE
# ============================================================
print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)
print("""
Summary:
1. The dataset was loaded and inspected.
2. Data type issues were fixed.
3. Categorical and numerical variables were analyzed.
4. Missing values and outliers were checked and handled.
5. New features were created through feature engineering.
6. Encoding and scaling steps were applied.
7. Multiple machine learning models were compared.
8. Best candidate models were tuned using GridSearchCV.
9. Final models were evaluated on the hold-out test set.
10. Visual outputs were saved into the images folder.

Business Insight:
This project helps a telecom company identify customers at risk of churn,
so retention strategies can be applied before losing those customers.
""")