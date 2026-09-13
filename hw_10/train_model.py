import pandas as pd
import joblib
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

df = pd.read_csv("realty_data.csv")

features = ["total_square", "rooms", "floor", "lat", "lon"]
target = "price"

X = df[features]
y = df[target]

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=features)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2024
)

model = ElasticNet(alpha=1, random_state=2024)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
print("R2:", r2_score(y_test, pred))

joblib.dump(model, "elastic_model.pkl")
print("Модель сохранена в elastic_model.pkl")