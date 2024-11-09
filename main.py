import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


dataset = pd.read_csv("msleep.csv")
subset = dataset.iloc[:, 6:11]
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(subset)
subset = imputer.transform(subset)
dataset.iloc[:, 6:11] = subset

X = pd.concat([dataset.iloc[:, 2:3], dataset.iloc[:, 6:11]],axis=1).values
y = dataset.iloc[:, 5]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(sparse_output=False),[0])],remainder="passthrough")
X = np.array(ct.fit_transform(X))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
error = np.abs(y_test.values - y_pred)
print(np.concatenate((y_test.values.reshape(len(y_test),1), y_pred.reshape(len(y_pred),1)), axis=1))

animal_names_in_test = dataset.loc[y_test.index,"name"]
df = pd.DataFrame({
    "Animal Names": animal_names_in_test,
    "Error": error
})
top_10_errors = df.nlargest(10,"Error")
plt.figure(figsize=(15,8))
plt.bar(x=df["Animal Names"],height=df["Error"])
plt.yscale("log")
plt.xlabel("Animals")
plt.ylabel("Error (Log Scale)")
plt.title("Top 10 Animals with Highest Prediction Errors")
plt.xticks(rotation=90)
plt.show()