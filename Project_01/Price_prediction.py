import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

houseData_train = pd.read_csv("train.csv")
houseData_test = pd.read_csv("test.csv")
print("Your preffered training Dataset: \n",houseData_train.head())
X_train= houseData_train[['LotArea','BedroomAbvGr','FullBath']]
y_train = houseData_train[['SalePrice']]

model = LinearRegression()
model.fit(X_train,y_train)
#Training completed

X_test= houseData_test[['LotArea','BedroomAbvGr','FullBath']]




y_pred_test = model.predict(X_test)

print("Predicted prices: \n",y_pred_test)
#mse = mean_squared_error(y_train,y_pred_test)
#print(f"Mean squared Error = {mse}")


plt.scatter(X_test['LotArea'],y_train['SalePrice'], color="blue" ,label = 'actual')
plt.scatter(X_test['LotArea'],y_pred_test,color="red", label="predicted")
plt.xlabel('Lot Area')
plt.ylabel('Prices')


plt.legend()
plt.show()




