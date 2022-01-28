import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from xgboost.core import _convert_ntree_limit

# Data collection and processing

df0 = pd.read_csv("D:\Python.vs\Data Set\calories_burnt\calories.csv")
df1 = pd.read_csv("D:\Python.vs\Data Set\calories_burnt\exercise.csv")

print(df0.head())
print(df1.head())

# Combaining the Data Frames 

df = pd.concat([df1, df0["Calories"]], axis = 1)
print(df.head())

# Converting the text data into numerical values

df.replace({"Gender": {"male" : 0, "female" : 1}}, inplace = True)
print(df.head())

# Knowing our Data Frame

print(df.isnull().sum())
print(df.describe())

# Visualization of our Data Frame

sns.set()

sns.catplot(x =  df["Gender"], data = df, kind = "count")
plt.show()

sns.displot(x = df["Age"], kde = True)
plt.show()

sns.displot(x = df["Weight"], kde = True)
plt.show()

# Correlation

print(df.corr())
plt.figure(figsize = (10, 10))
sns.heatmap(df, cbar=True, square=True, fmt=".1f",
            annot=True, annot_kws={'size': 8}, cmap="Blues")
plt.show()

# Saparating Features and Data

x = df.drop(columns = ["User_ID", "Calories"], axis = 1)
y =  df["Calories"]

# Train Test Split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)
print("Shapes: ", x.shape, x_train.shape, x_test.shape)

# Model Training

model = XGBRegressor()
model.fit(x_train, y_train)

# Evalution

test_prediction = model.predict(x_test)

# Mean Absolute Error

score_4 = metrics.mean_absolute_error(y_test, test_prediction)
print('Mean Absolute Error : ', score_4)

# Visualizing the actual Prices and predicted prices on train data

plt.scatter(y_test, test_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Preicted Price on testing data")
plt.show()















