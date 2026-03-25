# @ here i'm create a house predictor model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np



df = pd.read_csv("/content/drive/MyDrive/sample_data.csv")



#fix the dataset
df = df.dropna(subset=['price', 'house_size', 'bed', 'bath', 'brokered_by','acre_lot'])

#remove outliers
df = df[(df['price'] >= 100000.0) & (df['price'] <= 4000000.0)]
df = df[(df['house_size'] > 100) & (df['house_size'] < 20000)]

df = df.dropna(subset=['price'])
X = df[['state', 'city', 'brokered_by', 'zip_code', 'bed', 'bath', 'house_size', 'acre_lot']]
y = df['price']

#make plot
fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
x = df['price']
y = df['house_size']
ax.plot(x, x, label='linear')  
ax.plot(x, x**2, label='quadratic') 
ax.plot(x, x**3, label='cubic') 
ax.set_xlabel('x label') 
ax.set_ylabel('y label')  
ax.set_title("Simple Plot") 
ax.legend()

#to convert the data set into train test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# preprocess the data
cat_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='constant',fill_value='unknown')),
    ('encoding', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])
num_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean'))
])
preprocess = ColumnTransformer([
    ('cat_transformer', cat_transformer, ['state','city','brokered_by','zip_code']),
    ('num_transformer', num_transformer, ['bath','bed','house_size','acre_lot'])
])

model_pipeline = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model',RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42))
])

#train the data
y_train_log = np.log1p(y_train)
model_pipeline.fit(X_train, y_train_log)

y_pred_log = model_pipeline.predict(X_test)
y_pred = np.expm1(y_pred_log)
mse = mean_squared_error(y_test, y_pred)

#to print the model learning score
print(f"r2 score: {r2_score(y_test, y_pred):.2f}")
print(f"mse score: {mse:.2f}")
print(df.head())
