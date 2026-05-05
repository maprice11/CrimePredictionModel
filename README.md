# Predicting Criminal Behavior in Major US Cities

This final project in my Data Science Project/Portfolio Class was conducted as my cumulative understanding of my teachings under Belmont University's data science program. It ultimately aims to create a prediction model that collects demographic data of a city and produces a predicted violence rating of that city on a scale of 0 (No Violence) to 10 (Extremely Violent). Other statistical analyses were implemented to test the model's efficiency.

## Data Wrangling and Cleaning
<sub><sup> Detailed programming used for data wrangling and cleaning can be found under this repository's file: Final_Project_Dataframe_cleaning. </sup></sub> 

Data was collected from seven cities (Austin, San Francisco, Los Angeles, New York, Chicago, Seattle, and Detroit) that included demographic information from the United States Census website and crime data queried from open source government datasets that were available from each city's databases. 

### Data Collection

Demographic information was manually input into an Excel file. This file included the demographic breakdown of each city and included 31 demographic variables with information that included things like education, racial breakdown, poverty, and health factors of each city. 

Arrest data was collected from each city that collected every single arrest from 2018 - 2024. Every arrest date was reformatted to ensure that there was a consistent format for every city.

```
def sample_day(group):
  n = min(len(group), 1)
  return group.sample(n)
```

This function randomly selected one arrest per day to include in each city's dataframe. 

Columns for each city were renamed to ensure consistency and city name was added to the dataframes. After collecting all of the arrest data (one arrest per day, per city, from 2018-2024) for each individual city in a separate dataframe, all cities were concatenated into a final crime dataframe. 

### Fuzzy String NLP & Variable Creation

A majority of time spent wrangling the data was assigning arrests to specific crime groups. 

Using Fuzzy Strings, a form of Natural Language Processing, arrests were categorized into more generic crime categories. All arrests were passed through a basic fuzzy string function.

```
def match_crime(crime_type, threshold = 75):
  # Ensure crime_type is a string before processing
  crime_type_str = str(crime_type) if pd.notna(crime_type) else ''
  if not crime_type_str.strip(): # Handle empty or whitespace-only strings
      return 'Other'
  match, score = process.extractOne(crime_type_str, categories)
  return match if score >= threshold else 'Other'

crime_df['Crime Group'] = crime_df['Crime'].apply(match_crime)
```

The function takes all arrest information and assigns it to one of the following categories at a 75% similarity level (if the arrest shares 75% of a similar description as the category).

```
categories = ['Assault', 'Robbery', 'Homicide', 'Theft',
              'Burglary', 'Drug Offense', 'Sexual Assault', 'Vandalism', 'Trespassing', 'Battery']
```

After the first pass, the remaining arrest records were passed through a second, much more thorough, function.

```
# Expand your categories with synonyms and alternate terms
expanded_categories = {
    'Theft'          : ['Theft', 'Larceny', 'Shoplifting', 'Stealing',
                        'Pickpocket', 'Pilfering', 'Stolen', 'Purse snatching', 'Vehicle, Stolen', 'Carjacking', 'Bunco'],
    'Assault'        : ['Assault', 'Battery', 'Intimidation',
                        'Threatening', 'Menacing', 'Harassment', 'Strangulation', 'Kidnapping', 'Coercion',
                        'Breath/Circ', 'AGG ASLT', 'Injury disabled individual'],
    'Drug Offense'   : ['Drug', 'Narcotics', 'Controlled Substance',
                        'Possession', 'Distribution', 'Cannabis', 'Marijuana', 'Meth'],
    'Fraud'          : ['Fraud', 'Forgery', 'Embezzlement', 'Deception',
                        'Identity Theft', 'Scam', 'Gambling', 'Credit Card', 'False Pretenses', 'Bad Checks', 'Money Laundering',
                        'Records Falsify', 'Counterfeiting', 'Peddling', 'Breach of Computer Security'],
    'Weapons'        : ['Weapons', 'Firearm', 'Gun', 'Armed', 'Carrying', 'Shots',
                        'Concealed Carry'],
    'Vandalism'      : ['Vandalism', 'Damage', 'Graffiti', 'Destruction', 'Arson', 'Criminal Mischief'],
    'Trespassing'    : ['Trespassing', 'Unlawful Entry', 'Breaking', 'Trespass'],
    'Public Order'   : ['Disorderly', 'Disturbing Peace', 'Public Intoxication',
                        'DUI', 'Loitering', 'Impersonation', 'Driving', 'Contempt',
                        'No Contact', 'Family Offenses', 'Terroristic Threat', 'Prowler', 'Public Peace',
                        'Riot', 'Bail Jumping', 'Resisting Arrest', 'Perjury', 'Conspiracy', 'Family Disturbance', 'Obstructing Justice',
                        'Suspicious Occ', 'DWI', 'DOC', 'DOC discharge gun'],
    'Misdemeanor'    : ['Misdemeanor', 'Liquor Law', 'Criminal Mis', 'Moving Traffic', 'Failure to Yield', 'Sale School Grounds', 'False Report'],
    'Sexual Misconduct' : ['Sexual Misconduct', 'Sex', 'Trafficking', 'Sex Offender', 'Pornography', 'Prostitution', 'Obscenity', 'Lewd', 'LETTERS, LEWD',
                           'CSC 1st Degree'],
    'Sexual Assault' : ['Rape', 'Sodomy', 'Exposure', 'Fondling', 'Oral Copulation', 'Forcible Touching'],
    'Other'          : ['Other', 'Recovered Vehicle', 'Lost Property', 'Miscellaneous Investigation',
                        'Warrant', 'Missing Adult'],
    'Animal Cruelty'  : ['Animal Cruelty', 'Animal'],
    'Child-Related Crimes' : ['Child', 'Endangerment', 'Crm Agnst Chld', 'Pre Delinquency', 'Online Solicitation of a Minor'],
    'Homicide' : ['Murder', 'Manslaughter'],
    'Terrorism' : ['Bomb Scare'],
    'Uncategorized' : ['Unclassified']

}

def second_fuzzy_pass(crime_label, threshold=70):
    # This function expects crime_label to be a native Python string due to the .astype(str) call below.
    # Defensive checks are still useful for clarity but less critical with prior type enforcement.
    crime_label_str = str(crime_label)
    if not crime_label_str.strip():
        return 'Other'

    best_score = 0
    best_category = 'Uncategorized'

    for category, synonyms in expanded_categories.items():
        match, score = process.extractOne(crime_label_str, synonyms)
        if score > best_score:
            best_score = score
            best_category = category

    return best_category if best_score >= threshold else 'Uncategorized'

# Ensure the 'Crime' column is of string type before applying fuzzy matching
# This is crucial to prevent TypeErrors from fuzzy matching library's C++ bindings
crime_df['Crime'] = crime_df['Crime'].astype(str)

# Apply ONLY to rows currently labeled Other
crime_df.loc[others, 'Crime Group'] = (
    crime_df.loc[others, 'Crime'].apply(second_fuzzy_pass)
)
```

The second pass allowed for arrests to be assigned to crime categories based on specific synonyms and alternate terms for the crime. All further unknown crimes were assigned to Uncategorized. 

Finally, each crime group was assigned a violence rating out of 10, which were as follows:

```
    'Homicide'             : 10,
    'Sexual Assault'       : 9,
    'Terrorism'            : 9,
    'Child-Related Crimes' : 8,
    'Robbery'              : 8,
    'Sexual Misconduct'    : 7,
    'Assault'              : 6,
    'Battery'              : 6,
    'Weapons'              : 5,
    'Burglary'             : 4,
    'Trespassing'          : 3,
    'Drug Offense'         : 2,
    'Vandalism'            : 2,
    'Fraud'                : 1,
    'Theft'                : 1,
    'Public Order'         : 2,
    'Animal Cruelty'       : 2,
    'Misdemeanor'          : 1,
    'Other'                : 0,
    'Uncategorized'        : 0
```

The violence ratings were mapped onto the dataframe based on the crime group. 

<sub><sup> The use of Fuzzy Strings was assisted using Claude AI. </sup></sub> 

### Combining Datasets

The crime dataset was aggregated to be on a per year level, rather than per day, for each city. This allows for the demographic data to be more relevant in analyses, since it is the same data across all years. 

```
city_year = crime.groupby(['City', 'Year']).agg(
    Total_Arrests = ('Crime', 'count'),
    Avg_Violence_Score = ('Violence Rating', 'mean'),
    Violent_Crime_Count = ('Violence Rating', lambda x: (x >= 6).sum())
).reset_index()
```

Each city has a violent crime rating per year, which is an average of the randomly selected one arrest per day that was used in the original crime data frame. 

A new variable, Violent Crime Rate, was created to see what percentage of all crimes in a city each year were considered violent (average crime score is greater than 6).

The crime dataset was left-joined with the census data set into a final master dataframe. The violence score breakdowns of each city are reflected in Figure 1.

## Figure 1
<img width="1189" height="593" alt="Demographic Information" src="https://github.com/user-attachments/assets/0b785100-59e0-4422-8e39-f26a551b58c6" />

### Feature Engineering

Columns that were skewed/had extreme differences were logged.

```
#transforming skewed columns/extreme differences
log_cols = [
    'Population',
    'HousingUnits',
    'Households',
    'Asian',
    'Black or African American',
    'White',
    'Hispanic or Latino',
    'American Indian and Alaska Native',
    'Poverty (percent)',
    'Residential Mobility (percent)',
]
```
Feature variables that were included were all columns in the master dataset besides the following:

```
    'City',
    'Year',
    'Avg_Violence_Score',
    'Violent_Crime_Count',
    'Violent_Crime_Rate',
    'Total_Arrests',
    'Largest Housing Value',

```

Non-logged variables were also dropped from feature variables. 

All of the feature variables were scaled to standardize the data. 

## Data Analysis
<sub><sup> Detailed programming used for data analyses can be found under this repository's file: Dataset_Merging_and_Analyses. </sup></sub> 

The analyses that were conducted on the data were a Principal Component Analysis, Linear Regression, Ridge Regression, Lasso Regression, and a Random Forest Decision Tree. 

### Principal Component Analysis (PCA)

Because there were so many demographic variables (31 in total), it was necessary to boil down the variables to feature components that would explain at least 90% of the variance that the original model with all 31 variables would. A PCA was run to determine how many components would lead to this minimum variance. 

It was found that the PCA took 31 variables and found 3 components that accounted for 93.2% of the variance explained by the original model. 

```

PCA: 31 features -> 3 components
Variance explained: 93.2%
PC 1: 43.2%
PC 2: 35.0%
PC 3: 15.1%

```

The top three components that drove each component were:

```

PC1: log_Asian, Foreign-born (percent), Homeownership Rate(percent)
PC2: log_Black or African American, Graduate or Professional (percent), Education (percent)
PC3: Health (percent), Older Pop (percent), Age

```

Figure 2 describes the amount of variance explained by each component and the cumulative variance explained by the components. It also illustrates which specific cities are influenced most greatly by the first and second principal component. 


## Figure 2
<img width="1186" height="593" alt="Principal Component Analysis" src="https://github.com/user-attachments/assets/b0ba3900-4e8b-4661-bb35-ac58e26e708b" />

### Cross Validation

All of the following models were cross validated using 5-fold CV to ensure there was no overfitting. 

```

#cross validation setup (5 k fold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(name, model, X, y, cv):
  scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
  rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'))
  print(f"{name:<20} R^2: {scores.mean():.3f} +/- {scores.std():.3f} | "
        f"RMSE: {rmse_scores.mean():.3f} +/- {rmse_scores.std():.3f}")
  return scores

```

### Regression Models

The R^2 values of each model indicate how much variance is explained by each model, the closer the value is to 1, the stronger the model.

#### Linear Regression

```

Model Results (5-fold cross validation):
Linear Regression    R^2: 0.714 +/- 0.184 | RMSE: 0.262 +/- 0.065

Coefficients: ['PC1: -0.069', 'PC2: 0.115', 'PC3: 0.142']

```
PC3 is the most influential component in the model.

#### Ridge Regression

```
Ridge Regression     R^2: 0.714 +/- 0.181 | RMSE: 0.263 +/- 0.064
 Best alpha: 0.01

```
PCA already did the heavy lifting of reducing variables into components, so an alpha of 0.01 indicates that the ridge regression barely needs to penalize anything. 

#### Lasso Regression

```
Lasso Regression     R^2: 0.716 +/- 0.183 | RMSE: 0.261 +/- 0.065
 Best alpha: 0.0012
 Nonzero coefficients: 3/3 components kept

```

Utilizing lasso regression, it was found that all 3 components that were evaluated using PCA are necessary to the model and should not be dropped. 

Figure 3 describes the predicted violence score vs. actual violence score that is described by the Linear Regression Model. It also evaluates the predicted coefficient values for each model, with all thre models having basically exact coefficient values and PC3 being the component with the highest predictor value in the model.

## Figure 3
<img width="1189" height="593" alt="Regression Models" src="https://github.com/user-attachments/assets/e38b79ee-fb46-42fd-a188-d0b64369c0e4" />

## Prediction Model

### Random Forest Decision Tree

A random forest model was also included in analyzing model efficiency.

```
Random Forest        R^2: 0.718 +/- 0.127 | RMSE: 0.278 +/- 0.070

 Random Forest feature importances:
Component  Importance
      PC3    0.488858
      PC2    0.447998
      PC1    0.063144
```

It tells a similar story, in that PC3 is the most important component in the model, which is also illustrated in Figure 4. Figure 4 also analyzes all 4 models created. Since Random Forest has the highest R^2 value, and the least amount of standard error, the final prediction model was built with Random Forest. 

## Figure 4
<img width="1189" height="593" alt="Random Forest Models" src="https://github.com/user-attachments/assets/5c616ea6-fa3d-406d-b425-d4595cb32864" />

### Final Model

Users are able to adjust the city_predict dictionary to input any specific demographics. The dictionary for demographics on Nashville are as follows:

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

Natural Language Processing (Fuzzy String) - 2

Ridge/Lasso Regression - 1

Cross Validation - 0.5
