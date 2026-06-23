import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_excel("pima-data.xlsx")

X = df[['num_preg',
        'glucose_conc',
        'diastolic_bp',
        'skin',
        'insulin',
        'bmi',
        'diab_pred',
        'age']]

y = df['diabetes']

imputer = SimpleImputer(
    missing_values=0,
    strategy='mean'
)

X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(x_train, y_train)

joblib.dump(model,
            "models/diabetes_model.pkl")

joblib.dump(imputer,
            "models/imputer.pkl")

joblib.dump(scaler,
            "models/scaler.pkl")

print("Model Saved Successfully")