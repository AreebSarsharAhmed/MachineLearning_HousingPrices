#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported!")


# In[4]:


train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

train.head()


# In[5]:


# Basic structure of the dataset
train.info()


# In[6]:


# Summary statistics for numerical features
train.describe()


# In[7]:


# Check columns with missing values
train.isna().sum().sort_values(ascending=False).head(10)


# In[8]:


# Create a histogram to show how house prices are distributed
sns.histplot(train["SalePrice"], kde=True)  # Plot SalePrice with a smooth curve

# Add a title to the graph
plt.title("Distribution of Sale Prices")

# Label the x-axis
plt.xlabel("Sale Price")

# Label the y-axis
plt.ylabel("Count")

# Display the plot
plt.show()


# In[9]:


# Create a boxplot to identify outliers in house prices
sns.boxplot(x=train["SalePrice"])  # Draw boxplot for SalePrice

# Add a title to the plot
plt.title("Boxplot of Sale Prices")

# Label the x-axis
plt.xlabel("Sale Price")

# Display the plot
plt.show()


# In[10]:


# Select only numeric columns from the dataset
numeric_data = train.select_dtypes(include=[np.number])

# Calculate correlation between numeric features
corr_matrix = numeric_data.corr()

# Set the size of the figure
plt.figure(figsize=(10, 8))

# Draw a heatmap to visualize correlations
sns.heatmap(corr_matrix, cmap="coolwarm")

# Add a title to the heatmap
plt.title("Correlation Heatmap of Numeric Features")

# Display the heatmap
plt.show()


# In[11]:


# Create a scatter plot to show the relationship between house size and price
sns.scatterplot(x=train["GrLivArea"], y=train["SalePrice"])  # Plot living area vs price

# Add a title to the plot
plt.title("GrLivArea vs SalePrice")

# Label the x-axis
plt.xlabel("Above Ground Living Area (sq ft)")

# Label the y-axis
plt.ylabel("Sale Price")

# Display the plot
plt.show()


# In[12]:


# Separate input features (everything except SalePrice)
X = train.drop("SalePrice", axis=1)  # X contains all predictor variables

# Separate target variable (what we want to predict)
y = train["SalePrice"]  # y contains house prices

# Print the shape of X to check number of rows and columns
print("X shape:", X.shape)

# Print the shape of y to check number of target values
print("y shape:", y.shape)


# In[13]:


# Import function to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X,        # Input features
    y,        # Target variable
    test_size=0.2,   # 20% of data used for testing
    random_state=42  # Fixed number for reproducibility
)

# Print shapes to confirm the split
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[14]:


# Import NumPy for data type checking
import numpy as np

# Select all numeric columns from X
num_features = X.select_dtypes(include=[np.number]).columns

# Select all categorical (non-numeric) columns from X
cat_features = X.select_dtypes(exclude=[np.number]).columns

# Print how many numeric features we have
print("Number of numeric features:", len(num_features))

# Print how many categorical features we have
print("Number of categorical features:", len(cat_features))

# Show a few example numeric column names
print("Example numeric features:", list(num_features[:5]))

# Show a few example categorical column names
print("Example categorical features:", list(cat_features[:5]))


# In[15]:


# Import ColumnTransformer to apply different steps to different columns
from sklearn.compose import ColumnTransformer  # Helps process numeric and categorical columns separately

# Import Pipeline to run steps in order
from sklearn.pipeline import Pipeline  # Allows multiple preprocessing steps in sequence

# Import SimpleImputer to fill missing values
from sklearn.impute import SimpleImputer  # Fills missing (NaN) values

# Import OneHotEncoder to convert text categories into numbers
from sklearn.preprocessing import OneHotEncoder  # Converts categorical text into numeric columns

# Import StandardScaler to scale numeric values
from sklearn.preprocessing import StandardScaler  # Scales numeric features to similar ranges

# Create a pipeline for numeric columns (fill missing + scale)
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),  # Fill missing numeric values with the median
    ("scaler", StandardScaler())                    # Scale numeric values
])

# Create a pipeline for categorical columns (fill missing + one-hot encode)
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")), # Fill missing categorical values with the most common value
    ("onehot", OneHotEncoder(handle_unknown="ignore"))    # Convert categories to numbers and ignore unseen categories
])

# Combine numeric and categorical pipelines into one preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),  # Apply numeric steps to numeric columns
        ("cat", categorical_transformer, cat_features) # Apply categorical steps to categorical columns
    ]
)

# Display the preprocessor object (to confirm it was created)
preprocessor


# In[16]:


# Import Ridge regression model
from sklearn.linear_model import Ridge  # Linear regression with regularization

# Import metric function for error calculation
from sklearn.metrics import mean_squared_error  # Used to calculate RMSE

# Create a function to calculate RMSE (Root Mean Squared Error)
def rmse(y_true, y_pred):  # Define a function named rmse
    return np.sqrt(mean_squared_error(y_true, y_pred))  # RMSE = sqrt(MSE)

# Build a full pipeline: preprocessing + Ridge model
ridge_model = Pipeline(steps=[
    ("preprocessor", preprocessor),                 # Step 1: clean/encode/scale the data
    ("model", Ridge(alpha=1.0, solver="lsqr"))      # Step 2: Ridge model (lsqr works on your system)
])

# Train the Ridge model using training data
ridge_model.fit(X_train, y_train)  # Fit model on training set

# Predict house prices for the test set
y_pred_ridge = ridge_model.predict(X_test)  # Make predictions on unseen data

# Calculate RMSE for Ridge predictions
ridge_rmse = rmse(y_test, y_pred_ridge)  # Compute RMSE

# Print the Ridge RMSE result
print("Ridge RMSE:", ridge_rmse)  # Show the error number


# In[17]:


# Import Random Forest regression model
from sklearn.ensemble import RandomForestRegressor  # Tree-based ensemble model

# Build a full pipeline: preprocessing + Random Forest model
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),                 # Step 1: clean/encode/scale the data
    ("model", RandomForestRegressor(
        n_estimators=300,                           # Number of trees in the forest
        random_state=42,                            # Fixed number for reproducible results
        n_jobs=-1                                   # Use all CPU cores to speed up training
    ))
])

# Train the Random Forest model using training data
rf_model.fit(X_train, y_train)  # Fit the model on training data

# Predict house prices for the test set
y_pred_rf = rf_model.predict(X_test)  # Make predictions on unseen data

# Calculate RMSE for Random Forest predictions
rf_rmse = rmse(y_test, y_pred_rf)  # Compute RMSE

# Print the Random Forest RMSE result
print("Random Forest RMSE:", rf_rmse)  # Show the error number


# In[18]:


# Print Ridge model error for comparison
print("Ridge RMSE:", ridge_rmse)  # Baseline model error

# Print Random Forest model error for comparison
print("Random Forest RMSE:", rf_rmse)  # Improved model error


# In[19]:


# Import function to create learning curves
from sklearn.model_selection import learning_curve  # Helps analyze model performance vs data size

# Generate learning curve data
train_sizes, train_scores, test_scores = learning_curve(
    rf_model,                     # Use the Random Forest model
    X_train,                      # Training features
    y_train,                      # Training target
    cv=5,                         # Use 5-fold cross-validation
    scoring="neg_root_mean_squared_error",  # Use RMSE as the metric
    train_sizes=np.linspace(0.1, 1.0, 5),   # Train on 10% to 100% of data
    n_jobs=-1                     # Use all CPU cores
)

# Convert negative RMSE values to positive
train_rmse = -train_scores.mean(axis=1)  # Average training RMSE
test_rmse = -test_scores.mean(axis=1)    # Average validation RMSE

# Plot the learning curve
plt.figure(figsize=(8, 6))               # Set figure size
plt.plot(train_sizes, train_rmse, marker="o", label="Training RMSE")   # Plot training error
plt.plot(train_sizes, test_rmse, marker="o", label="Validation RMSE")  # Plot validation error

# Label the plot
plt.xlabel("Training Set Size")           # X-axis label
plt.ylabel("RMSE")                        # Y-axis label
plt.title("Learning Curve - Random Forest")  # Plot title
plt.legend()                              # Show legend

# Display the plot
plt.show()


# In[ ]:




