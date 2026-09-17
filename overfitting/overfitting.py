import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(SCRIPT_DIR, "data_1.csv"))


print(df.head())
print(df.shape)
print(df.columns)
print(df.dtypes)




X = df.drop(columns=["gia_nha_trieu"]) #Lấy tất cả các cột ngoại trừ date và Appliances làm dữ liệu đầu vào.

y = df["gia_nha_trieu"]


# chia du lieu de train, test

X_train, X_test, Y_train, Y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

#train model

model = LinearRegression()
model.fit(X_train, Y_train)

# du doan 

Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)


# danh gia mo hinh
train_mae = mean_absolute_error(
    Y_train,
    Y_train_pred
)

test_mae = mean_absolute_error(
    Y_test,
    Y_test_pred
)

print("Train MAE:", train_mae)
print("Test MAE:", test_mae)

train_r2 = r2_score(Y_train, Y_train_pred)
test_r2 = r2_score(Y_test, Y_test_pred)

print("Train R²:", train_r2)
print("Test R²:", test_r2)

# bị overfiting

# Train MAE: 1034.864788759842
# Test MAE: 2648.62461937323
# Train R²: 0.539972920754584
# Test R²: -0.71545751037204

