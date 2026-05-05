# Predicting Criminal Behavior in Major US Cities

This project ultimately aims to create a prediction model that collects demographic data of a city and produces a predicted violence rating of a city on a scale of 0 (No Violence) to 10 (Extremely Violent). Other statistical analyses were implemented to test the model's efficiency.

## Data Wrangling and Cleaning

### Data Collection

### Fuzzy String NLP

### Variable Creation

### Combining Datasets

### Feature Engineering

<img width="1189" height="593" alt="Demographic Information" src="https://github.com/user-attachments/assets/0b785100-59e0-4422-8e39-f26a551b58c6" />

## Data Analysis

### Principal Component Analysis (PCA)

<img width="1186" height="593" alt="Principal Component Analysis" src="https://github.com/user-attachments/assets/b0ba3900-4e8b-4661-bb35-ac58e26e708b" />

### Cross Validation

### Regression Models

#### Linear Regression

#### Ridge Regression

#### Lasso Regression

<img width="1189" height="593" alt="Regression Models" src="https://github.com/user-attachments/assets/e38b79ee-fb46-42fd-a188-d0b64369c0e4" />

## Prediction Model

### Random Forest Decision Tree

<img width="1189" height="593" alt="Random Forest Models" src="https://github.com/user-attachments/assets/5c616ea6-fa3d-406d-b425-d4595cb32864" />

The final prediction model was built using a Random Forest Decision Tree. Users are able to adjust the city_predict dictionary to input any specific demographics. The dictionary for demographics on Nashville are as follows:

```
city_predict = {
    'Population' : 698447,
    'Income (dollars)' : 80090,
    'Education (percent)' : 50.7,
    'Employment (percent)' : 72.2,
    'HousingUnits' : 369824,
    'Health (percent)' : 12.9,
    'Households' : 331090,
    'Age' : 34.6,
    'Language (percent)' : 20.2,
    'Foreign-born (percent)' : 15.6,
    'Naturalized (percent)' : 37.5,
    'Not Naturalzied (percent)' : 62.5,
    'Older Pop (percent)' : 12.9,
    'Residential Mobility (percent)' : 4.7,
    'Poverty (percent)' : 12.1,
    'High School (percent)' : 19.6,
    "Bachelor's (percent)" : 30.9,
    'Graduate or Professional (percent)' : 19.9,
    'K-12 (percent)' : 60.5,
    'Commuting' : 25.2,
    'Hours Worked' : 38.8,
    'Rent' : 1669,
    'Homeownership Rate (percent)' : 51.8,
    'Disability (percent)' : 10.3,
    'Marital Status (percent )' : 43.4,
    'American Indian and Alaska Native' : 3839,
    'Asian' : 27375,
    'Black or African American' : 169349,
    'Hispanic or Latino' : 96349,
    'Native Hawaiian and Other Pacific Islander' : 318,
    'White' : 380838
 }
```

The final prediction function creation:

```
rf_final = RandomForestRegressor(n_estimators = 200, max_depth=3,
                                 min_samples_leaf=3, random_state=42)
rf_final.fit(X_pca, y)

def predict_violence(demographic_input: dict) -> float:
  input_df = pd.DataFrame([demographic_input])
  for col in log_cols:
    input_df[f'log_{col}'] = np.log1p(input_df[col])
    input_df.drop(columns=[col], inplace=True)
  input_df = input_df.reindex(columns=features, fill_value = 0)

  input_scaled = scaler.transform(input_df)
  input_pca = pca.transform(input_scaled)

  prediction = rf_final.predict(input_pca)[0]
  return round(prediction, 2)
```

The function first intakes the a demographic dictionary, then log transforms the assigned columns. Then, it ensures that the input columns of the data dictionary match the feature order of the model. The input data is then scaled and transformed using Principal Component Analysis. The prediction is created using the Random Forest final prediction model and the input data. 

The final lines of code that will present a visually appealing Violent Crime Rating for a specific city:

```
predicted_score = predict_violence(city_predict)
print(f"  Predicted Violence Score: {predicted_score} / 10")
print("\n  Scale reference:")
print("  0-2  → Very low violence")
print("  2-4  → Low violence")
print("  4-6  → Moderate violence")
print("  6-8  → High violence")
print("  8-10 → Very high violence")

```

Output for the demographics of Nashville:

```
 Predicted Violence Score: 2.58 / 10

  Scale reference:
  0-2  → Very low violence
  2-4  → Low violence
  4-6  → Moderate violence
  6-8  → High violence
  8-10 → Very high violence
```

## Advanced Topics

Linear Model - 0.5

Random Forest - 1

Principal Component Analysis (PCA) - 1

Combining Datasets - 0.5

Feature Selection/Engineering - 0.5

Bootstrap Confidence Intervals - 1.5

Natural Language Processing (Fuzzy String) - 2

Ridge/Lasso Regression - 1

Cross Validation - 0.5



```
Give examples
```

