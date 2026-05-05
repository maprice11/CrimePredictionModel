# Predicting Criminal Behavior in Major US Cities

This project ultimately aims to create a prediction model that collects demographic data of a city and produces a predicted violence rating of a city on a scale of 0 (No Violence) to 10 (Extremely Violent). Other statistical analyses were implemented to test the model's efficiency.

## Data Wrangling and Cleaning

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
