import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
sns.set(style="whitegrid")
# %matplotlib inline

# Loading train and test dataset

train=pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


data_dict=pd.read_csv("data_dictionary.csv")
sample_submission=pd.read_csv("sample_submission.csv")


# display(train.head())
# print(f"Train shape: {train.shape}")

# display(test.head())
# print(f"Test shape: {test.shape}")

data_dict.head()

# Filling Missing values

import pandas as pd


file_path = 'train.csv'
df = pd.read_csv(file_path)


missing_values = df.isnull().sum()


print("Missing Values in Each Column:")
print(missing_values)


total_missing = missing_values.sum()
print(f"\nTotal Missing Values in the DataFrame: {total_missing}")


missing_columns = missing_values[missing_values > 0]
print("\nColumns with Missing Values:")
print(missing_columns)


# Missing Values Count in Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('train.csv')


missing_counts = df.isnull().sum()


missing_counts_scaled = missing_counts / 100


plt.figure(figsize=(30, 10))
missing_counts_scaled.plot(kind='bar', color='skyblue')
plt.title('Missing Values Count in Dataset (Bar Chart)')
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values (in Hundreds)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()



# This next function is particularly useful during **exploratory data analysis**, as it helps identify data distributions, **outliers**, and **missing values**, facilitating informed decisions about data cleaning and preprocessing.

def calculate_stats(data, columns):
    if isinstance(columns, str):
        columns = [columns]

    stats = []
    for col in columns:
        if data[col].dtype in ['object', 'category']:
            counts = data[col].value_counts(dropna=False, sort=False)
            percents = data[col].value_counts(normalize=True, dropna=False, sort=False) * 100
            formatted = counts.astype(str) + ' (' + percents.round(2).astype(str) + '%)'
            stats_col = pd.DataFrame({'count (%)': formatted})
            stats.append(stats_col)
        else:
            stats_col = data[col].describe().to_frame().transpose()
            stats_col['missing'] = data[col].isnull().sum()
            stats_col.index.name = col
            stats.append(stats_col)

    return pd.concat(stats, axis=0)

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('train.csv')
data = {
    'Column A': [1, 2, 3, 4, 5],
    'Column B': ['A', 'B', 'C', 'D', 'E'],
    'Column C': [1.1, 2.2, 3.3, 4.4, 5.5],
    'Column D': ['X', 'Y', 'Z', 'A', 'B'],
    'Column E': [100, 200, 300, 400, 500],
}
df = pd.DataFrame(data)
type_counts = {
    'int': 0,
    'float': 0,
    'str': 0,
    'char': 0
}
for column in df.columns:
    if pd.api.types.is_integer_dtype(df[column]):
        type_counts['int'] += 1
    elif pd.api.types.is_float_dtype(df[column]):
        type_counts['float'] += 1
    elif pd.api.types.is_string_dtype(df[column]):
        type_counts['str'] += 1

        type_counts['char'] += df[column].apply(lambda x: len(x) == 1).sum()


type_distribution = pd.DataFrame(list(type_counts.items()), columns=['Type', 'Count'])


plt.figure(figsize=(10, 5))
plt.pie(type_distribution['Count'], labels=type_distribution['Type'], autopct='%1.1f%%', startangle=90)
plt.title('Data Type Distribution in Dataset (Pie Chart)')
plt.axis('equal')
plt.show()


# **Demographics**

vc = train['Basic_Demos-Enroll_Season'].value_counts()
plt.pie(vc.values, labels=vc.index) # Use values and index properties
plt.title('Season of enrollment')
plt.show()

vc = train['Basic_Demos-Sex'].value_counts()
plt.pie(vc, labels=['boys', 'girls'])  # Use vc directly
plt.title('Sex of participant')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'train' is your DataFrame containing the data
# Replace 'train' with the actual name of your DataFrame if it's different

# If 'supervised_usable' was meant to be 'train', replace the following line:
# corr_matrix = supervised_usable.select([ ... ])
# with:
corr_matrix = train[[
    'PCIAT-PCIAT_Total', 'Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-BMI',
    'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
    'Physical-Diastolic_BP', 'Physical-Systolic_BP', 'Physical-HeartRate',
    'PreInt_EduHx-computerinternet_hoursday', 'SDS-SDS_Total_T', 'PAQ_A-PAQ_A_Total',
    'PAQ_C-PAQ_C_Total', 'Fitness_Endurance-Max_Stage', 'Fitness_Endurance-Time_Mins','Fitness_Endurance-Time_Sec',
    'FGC-FGC_CU', 'FGC-FGC_GSND','FGC-FGC_GSD','FGC-FGC_PU','FGC-FGC_SRL','FGC-FGC_SRR','FGC-FGC_TL','BIA-BIA_Activity_Level_num',
    'BIA-BIA_BMC', 'BIA-BIA_BMI', 'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM',
    'BIA-BIA_FFMI','BIA-BIA_FMI', 'BIA-BIA_Fat','BIA-BIA_Frame_num','BIA-BIA_ICW','BIA-BIA_LDM','BIA-BIA_LST','BIA-BIA_SMM','BIA-BIA_TBW'
    # Add other relevant columns
]].corr()

sii_corr = corr_matrix['PCIAT-PCIAT_Total'].drop('PCIAT-PCIAT_Total')
filtered_corr = sii_corr[(sii_corr > 0.1) | (sii_corr < -0.1)]

print(filtered_corr)

plt.figure(figsize=(8, 6))
filtered_corr.sort_values().plot(kind='barh', color='coral')
plt.title('Features with Correlation > 0.1 or < -0.1 with PCIAT-PCIAT_Total')
plt.xlabel('Correlation coefficient')
plt.ylabel('Features')
plt.show()



fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Season of Enrollment
season_counts = train['Basic_Demos-Enroll_Season'].value_counts(dropna=False)

axes[0].pie(
    season_counts, labels=season_counts.index,
    autopct='%1.1f%%', startangle=90,
    colors=sns.color_palette("Set3")
)
axes[0].set_title('Season of Enrollment')
axes[0].axis('equal')

# Age Distribution by Sex
sns.histplot(data=train, x='Basic_Demos-Age',
    hue='Basic_Demos-Sex', multiple='dodge',
    palette="Set2", bins=20, ax=axes[1]
)
axes[1].set_title('Age Distribution by Sex')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()


# **Child's global assesment scale**

data = train[train['CGAS-CGAS_Score'].notnull()]
age_range = data['Basic_Demos-Age']
print(
    f"Age range for participants with CGAS-CGAS_Score data:"
    f" {age_range.min()} - {age_range.max()} years"
)

calculate_stats(train, 'CGAS-CGAS_Score')

plt.figure(figsize=(12, 5))

# CGAS-Season
plt.subplot(1, 2, 1)
cgas_season_counts = train['CGAS-Season'].value_counts(normalize=True)
plt.pie(
    cgas_season_counts,
    labels=cgas_season_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette("Set3")
)
plt.title('CGAS-Season')
plt.axis('equal')
# CGAS-CGAS_Score without outliers (score == 999)
plt.subplot(1, 2, 2)
sns.histplot(
    train['CGAS-CGAS_Score'].dropna(),
    bins=20, kde=True
)
plt.title('CGAS-CGAS_Score (Without Outlier)')
plt.xlabel('CGAS Score')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

bins = np.arange(0, 101, 10)
labels = [
    "1-10: Needs constant supervision (24 hour care)",
    "11-20: Needs considerable supervision",
    "21-30: Unable to function in almost all areas",
    "31-40: Major impairment in functioning in several areas",
    "41-50: Moderate degree of interference in functioning",
    "51-60: Variable functioning with sporadic difficulties",
    "61-70: Some difficulty in a single area",
    "71-80: No more than slight impairment in functioning",
    "81-90: Good functioning in all areas",
    "91-100: Superior functioning"
]

train['CGAS_Score_Bin'] = pd.cut(train['CGAS-CGAS_Score'], bins=bins, labels=labels
)

counts = train['CGAS_Score_Bin'].value_counts().reindex(labels)
prop = (counts / counts.sum() * 100).round(1)
count_prop_labels = counts.astype(str) + " (" + prop.astype(str) + "%)"

plt.figure(figsize=(18, 6))
bars = plt.barh(labels, counts)
plt.xlabel('Count')
plt.title('CGAS Score Distribution')

for bar, label in zip(bars, count_prop_labels):
    plt.text(
        bar.get_width(), bar.get_y() + bar.get_height() / 2, label, va='center'
    )
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

score_min_max = train.groupby('sii')['CGAS-CGAS_Score'].agg(['min', 'max'])
score_min_max = score_min_max.rename(
    columns={'min': 'Minimum CGAS Score', 'max': 'Maximum CGAS Score'}
)
score_min_max

# **Sleep disturbance scale**

plt.figure(figsize=(18, 5))

# SDS-Season (Pie Chart)
plt.subplot(1, 3, 1)
sds_season_counts = train['SDS-Season'].value_counts(normalize=True)
plt.pie(
    sds_season_counts,
    labels=sds_season_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette("Set3")
)
plt.title('SDS-Season')
# SDS-SDS_Total_Raw
plt.subplot(1, 3, 2)
sns.histplot(train['SDS-SDS_Total_Raw'].dropna(), bins=20, kde=True)
plt.title('SDS-SDS_Total_Raw')
plt.xlabel('Value')

# SDS-SDS_Total_T
plt.subplot(1, 3, 3)
sns.histplot(train['SDS-SDS_Total_T'].dropna(), bins=20, kde=True)
plt.title('SDS-SDS_Total_T')
plt.xlabel('Value')

plt.tight_layout()
plt.show()


# **Target variables and internet use**

### Visualizes the distribution of the Severity Impairment Index (SII)

sii_counts = train['sii'].value_counts().reset_index()
total = sii_counts['count'].sum()
sii_counts['percentage'] = (sii_counts['count'] / total) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))


sns.barplot(x='sii', y='count', data=sii_counts, palette='Blues_d', ax=axes[0])
axes[0].set_title('Distribution of Severity Impairment Index (sii)', fontsize=14)
for p in axes[0].patches:
    height = p.get_height()
    percentage = sii_counts.loc[sii_counts['count'] == height, 'percentage'].values[0]
    axes[0].text(p.get_x() + p.get_width() / 2,
        height + 5, f'{int(height)} ({percentage:.1f}%)',
        ha="center", fontsize=12
    )


sns.histplot(train['PCIAT-PCIAT_Total'].dropna(), bins=20, ax=axes[1])
axes[1].set_title('Distribution of PCIAT_Total', fontsize=14)
axes[1].set_xlabel('PCIAT_Total for Complete PCIAT Responses')

plt.tight_layout()
plt.show()


# **Parent-Child Internet Addiction Test**, which was not test,csv

train_cols = set(train.columns)
test_cols = set(test.columns)
columns_not_in_test = sorted(list(train_cols - test_cols))
data_dict[data_dict['Field'].isin(columns_not_in_test)]

# The next code calculates the **minimum** and **maximum** values of the **PCIAT-PCIAT_Total** column in the train DataFrame, grouped by unique values of the sii column, and organizes the results in a structured format. First, the groupby('**sii**') function groups the data based on the unique values in the sii column, essentially creating subsets of data for each unique sii value

pciat_min_max = train.groupby('sii')['PCIAT-PCIAT_Total'].agg(['min', 'max'])
pciat_min_max = pciat_min_max.rename(
    columns={'min': 'Minimum PCIAT total Score', 'max': 'Maximum total PCIAT Score'}
)
pciat_min_max


data_dict[data_dict['Field'] == 'PCIAT-PCIAT_Total']['Value Labels'].iloc[0]


# We perform **data filtering** and **visualization** using the pandas library. First, a new DataFrame named train_with_sii is created by filtering rows from the train DataFrame where the '**sii**' column is not null, using the notna() method and isna(). This approach is useful for both filtering and visually identifying missing data in a DataFrame

# PCIAT calculation range from 01 to 20

# We perform data filtering and visualization using the pandas library.
# First, a new DataFrame named train_with_sii is created by filtering rows from
# the train DataFrame where the 'sii' column is not null, using the notna() method and isna().
# This approach is useful for both filtering and visually identifying missing data in a DataFrame
train_with_sii = train[train['sii'].notna()]  # Create the train_with_sii DataFrame

PCIAT_cols = [f'PCIAT-PCIAT_{i+1:02d}' for i in range(20)]
recalc_total_score = train_with_sii[PCIAT_cols].sum(
    axis=1, skipna=True
)
(recalc_total_score == train_with_sii['PCIAT-PCIAT_Total']).all()

# Data percentage for the age group:"Children (5-12)," "Adolescents (13-18)," and "Adults (19-22).

train['Age Group'] = pd.cut(
    train['Basic_Demos-Age'],
    bins=[4, 12, 18, 22],
    labels=['Children (5-12)', 'Adolescents (13-18)', 'Adults (19-22)']
)
calculate_stats(train, 'Age Group')


# Data percentage of Gender where 0 as 'Male' and 1 as 'Female'

sex_map = {0: 'Male', 1: 'Female'}
train['Basic_Demos-Sex'] = train['Basic_Demos-Sex'].map(sex_map)
calculate_stats(train, 'Basic_Demos-Sex')

# Grouping the data by Age Group and sii.

stats = train.groupby(['Age Group', 'sii']).size().unstack(fill_value=0)
fig, axes = plt.subplots(1, len(stats), figsize=(18, 5))

for i, age_group in enumerate(stats.index):
    group_counts = stats.loc[age_group] / stats.loc[age_group].sum()
    axes[i].pie(
        group_counts, labels=group_counts.index, autopct='%1.1f%%',
        startangle=90, colors=sns.color_palette("Set3"),
        labeldistance=1.05, pctdistance=0.80
    )
    axes[i].set_title(f'SII Distribution for {age_group}')
    axes[i].axis('equal')
    plt.tight_layout()
plt.show()


stats = train.groupby(['Age Group', 'sii']).size().unstack(fill_value=0)
stats_prop = stats.div(stats.sum(axis=1), axis=0) * 100

stats = stats.astype(str) +' (' + stats_prop.round(1).astype(str) + '%)'
stats

# Seasonal Data Distribution

season_columns = [col for col in train.columns if 'Season' in col]
season_df = train[season_columns]
season_df


data = train[train['PreInt_EduHx-computerinternet_hoursday'].notna()]
age_range = data['Basic_Demos-Age']
print(
    f"Age range for participants with measured PreInt_EduHx-computerinternet_hoursday data:"
    f" {age_range.min()} - {age_range.max()} years"
)

train['PreInt_EduHx-computerinternet_hoursday'].unique()

# Time ranges in increasing order of hours

# Assuming 'PreInt_EduHx-computerinternet_hoursday' represents internet use
# Create 'internet_use_encoded' based on the existing column
train['internet_use_encoded'] = pd.cut(
    train['PreInt_EduHx-computerinternet_hoursday'],
    bins=[-1, 1, 3, 6, float('inf')],  # Adjust bins as needed
    labels=['0-1 hours', '1-3 hours', '3-6 hours', '6+ hours'],
    right=False  # To include the lower bound in each bin
)

# Now you can proceed with the countplot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax1 = sns.countplot(x='internet_use_encoded', data=train, palette="Set3", ax=axes[0])
axes[0].set_title('Distribution of Hours of Internet Use')
axes[0].set_xlabel('Hours per Day Group')
axes[0].set_ylabel('Count')

total = len(train['internet_use_encoded'])
for p in ax1.patches:
    count = int(p.get_height())
    percentage = '{:.1f}%'.format(100 * count / total)
    ax1.annotate(f'{count} ({percentage})', (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5),
                 textcoords='offset points')


sns.boxplot(y=train['Basic_Demos-Age'], x=train['internet_use_encoded'], ax=axes[1], palette="Set3")
axes[1].set_title('Hours of Internet Use by Age')
axes[1].set_ylabel('Age')
axes[1].set_xlabel('Hours per Day Group')


sns.boxplot(y='PreInt_EduHx-computerinternet_hoursday', x='Age Group', data=train, ax=axes[2], palette="Set3")
axes[2].set_title('Internet Hours by Age Group')
axes[2].set_ylabel('Hours per Day (Numeric)')
axes[2].set_xlabel('Age Group')

plt.tight_layout()
plt.show()


# **Artificial Neural Network**



# *  This trains a Artificial neural network model for binary classification using features such as age, heart rate, and BMI topredict the target variable 'sii'
# *  The data is preprocessed by normalizing the features, splitting it into training and test sets, andthen evaluating the model's accuracy after training.





import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense


data = pd.read_csv('train.csv')


X = data[['Basic_Demos-Age', 'Physical-HeartRate', 'Physical-BMI']].values
y = data['sii'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy[1] * 100:.2f}%')


predictions = model.predict(X_test)



# *   This model trains a **neural network model** to predict the daily hours spent on computers or the internet (PreInt_EduHx- computerinternet_hoursday) based on features like age, heart rate, and BMI
# *   The model is built, trained, and evaluated using abinary classification approach, with accuracy reported after testing



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense


data = pd.read_csv('train.csv')



X = data[['Basic_Demos-Age', 'Physical-HeartRate', 'Physical-BMI']].values
y = data['PreInt_EduHx-computerinternet_hoursday'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy[1] * 100:.2f}%')


predictions = model.predict(X_test)



# *   This categorizes individuals into low, mild, or severe risk levels based on their 'sii' score and uses an artificial neural network to predict these risk levels from features like age, heart rate, and BMI.
# *   The model is trained with one-hot encoded labels and then used to predict the risk levels for children aged 5-12, displaying the results with the corresponding risk categories.



import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


data = pd.read_csv('train.csv')



def categorize_risk(sii):
    if sii < 1:
        return 0
    elif 1 <= sii < 2:
        return 1
    else:
        return 2


data['risk_level'] = data['sii'].apply(categorize_risk)


X = data[['Basic_Demos-Age', 'Physical-HeartRate', 'Physical-BMI']].values
y = data['risk_level'].values


y = to_categorical(y, num_classes=3)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy[1] * 100:.2f}%')


age_range_data = data[(data['Basic_Demos-Age'] >= 5) & (data['Basic_Demos-Age'] <= 12)]


age_range_X = age_range_data[['Basic_Demos-Age', 'Physical-HeartRate', 'Physical-BMI']].values


age_range_X = scaler.transform(age_range_X)


predictions = model.predict(age_range_X)


predicted_classes = np.argmax(predictions, axis=1)


class_labels = {0: 'Low', 1: 'Mild', 2: 'Severe'}
predicted_labels = [class_labels[i] for i in predicted_classes]


for i in range(len(age_range_data)):
    age = age_range_data['Basic_Demos-Age'].iloc[i]
    heart_rate = age_range_data['Physical-HeartRate'].iloc[i]
    bmi = age_range_data['Physical-BMI'].iloc[i]
    prediction = predicted_labels[i]
    print(f"Age: {age}, Heart Rate: {heart_rate}, BMI: {bmi} -> Predicted Risk: {prediction}")




# *  This preprocesses a dataset by filling missing values, categorizing the target variable ('sii') into low, mild, and severe risk levels, and training an **Artificial Neural Network (ANN)** model to predict these risk levels based on features like age, sex, BMI, and others.
# *  It evaluates the model's performance, visualizes training history, and makes predictions for individuals aged 5-12, displaying their predicted risk levels along with a bar chart of the predicted risk distribution.



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


print("Data types in training data:")
print(train_data.dtypes)
print("\nAvailable columns in training data:")
print(train_data.columns.tolist())
print("\nAvailable columns in test data:")
print(test_data.columns.tolist())


numerical_cols = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
train_data[numerical_cols] = train_data[numerical_cols].fillna(train_data[numerical_cols].mean())


categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    train_data[col].fillna(train_data[col].mode()[0], inplace=True)


print("Missing values in training data after imputation:")
print(train_data.isnull().sum())


def categorize_risk(sii):
    if sii < 1:
        return 0
    elif 1 <= sii < 2:
        return 1
    else:
        return 2


train_data['risk_level'] = train_data['sii'].apply(categorize_risk)


features = [
    'Basic_Demos-Age',
    'Basic_Demos-Sex',
    'Physical-BMI',
    'Fitness_Endurance-Max_Stage',
    'FGC-FGC_CU',
    'BIA-BIA_BMI',
    'PAQ_A-PAQ_A_Total'
]


available_features = [feature for feature in features if feature in train_data.columns]
print("\nAvailable features for training:")
print(available_features)


X = train_data[available_features]
y = train_data['risk_level'].values


y = to_categorical(y, num_classes=3)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numerical_cols = [col for col in available_features if col not in categorical_cols]
categorical_cols = [col for col in categorical_cols if col in available_features]


numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numerical_cols),
                  ('cat', categorical_transformer, categorical_cols)],
    remainder='drop'
)


X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


print(f"Shape of processed training data: {X_train_processed.shape}")


model = keras.Sequential([
    layers.Input(shape=(X_train_processed.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(X_train_processed, y_train, epochs=50, batch_size=32, validation_data=(X_test_processed, y_test))


accuracy = model.evaluate(X_test_processed, y_test)
print(f'Test accuracy: {accuracy[1] * 100:.2f}%')


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


age_range_data = train_data[(train_data['Basic_Demos-Age'] >= 5) & (train_data['Basic_Demos-Age'] <= 12)]


age_range_X = age_range_data[available_features]


age_range_X_processed = preprocessor.transform(age_range_X)


predictions = model.predict(age_range_X_processed)


predicted_classes = np.argmax(predictions, axis=1)


class_labels = {0: 'Low', 1: 'Mild', 2: 'Severe'}
predicted_labels = [class_labels[i] for i in predicted_classes]


for i in range(len(age_range_data)):
    age = age_range_data['Basic_Demos-Age'].iloc[i]
    bmi = age_range_data['Physical-BMI'].iloc[i]
    prediction = predicted_labels[i]
    print(f"Age: {age}, BMI: {bmi} -> Predicted Risk: {prediction}")


risk_counts = {
    'Low': predicted_labels.count('Low'),
    'Mild': predicted_labels.count('Mild'),
    'Severe': predicted_labels.count('Severe')
}


plt.figure(figsize=(5, 2))
plt.bar(risk_counts.keys(), risk_counts.values(), color=['green', 'yellow', 'red'])
plt.title('Distribution of Predicted Risk Levels (Age 5-12)')
plt.xlabel('Risk Level')
plt.ylabel('Count')
plt.show()


#  The model is trained and evaluated, and the predictions are compared to actual risk levels through a visualization that shows the distribution of actual vs predictedrisk levels using a count plot.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


numerical_cols = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
train_data[numerical_cols] = train_data[numerical_cols].fillna(train_data[numerical_cols].mean())


categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    train_data[col].fillna(train_data[col].mode()[0], inplace=True)


def categorize_risk(sii):
    if sii < 1:
        return 0
    elif 1 <= sii < 2:
        return 1
    else:
        return 2


train_data['risk_level'] = train_data['sii'].apply(categorize_risk)


features = [
    'Basic_Demos-Age',
    'Basic_Demos-Sex',
    'Physical-BMI',
    'Fitness_Endurance-Max_Stage',
    'FGC-FGC_CU',
    'BIA-BIA_BMI',
    'PAQ_A-PAQ_A_Total'
]


X = train_data[features]
y = train_data['risk_level'].values


y = to_categorical(y, num_classes=3)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numerical_cols = [col for col in features if col not in categorical_cols]
categorical_cols = [col for col in categorical_cols if col in features]


numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='drop'
)


X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


model = keras.Sequential([
    layers.Input(shape=(X_train_processed.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train_processed, y_train, epochs=50, batch_size=32, validation_data=(X_test_processed, y_test))


accuracy = model.evaluate(X_test_processed, y_test)
print(f'Test accuracy: {accuracy[1] * 100:.2f}%')


predictions = model.predict(X_test_processed)
predicted_classes = np.argmax(predictions, axis=1)




results_df = pd.DataFrame({
    'Actual': np.argmax(y_test, axis=1),
    'Predicted': predicted_classes,
})


plt.figure(figsize=(12, 6))
sns.countplot(data=results_df.melt(), x='value', hue='variable', palette='Set1')
plt.title('Distribution of Actual vs Predicted Risk Levels')
plt.xlabel('Risk Level')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1, 2], labels=['Low', 'Mild', 'Severe'])
plt.legend(title='Type', labels=['Actual', 'Predicted'])
plt.show()



# Distribution of risk levels

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


numerical_cols = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
train_data[numerical_cols] = train_data[numerical_cols].fillna(train_data[numerical_cols].mean())


categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    train_data[col].fillna(train_data[col].mode()[0], inplace=True)


def categorize_risk(sii):
    if sii < 1:
        return 0
    elif 1 <= sii < 2:
        return 1
    else:
        return 2


train_data['risk_level'] = train_data['sii'].apply(categorize_risk)


risk_counts = train_data['risk_level'].value_counts()


labels = ['Low Risk', 'Mild Risk', 'Severe Risk']
sizes = [risk_counts.get(0, 0), risk_counts.get(1, 0), risk_counts.get(2, 0)]
colors = ['lightblue', 'lightgreen', 'salmon']

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Risk Levels Among Individuals')
plt.axis('equal')
plt.show()

# ID's with SII levels mild and severe

import pandas as pd


train_data = pd.read_csv('train.csv')


def categorize_risk(sii):
    if sii < 1:
        return 0
    elif 1 <= sii < 2:
        return 1
    else:
        return 2


train_data['risk_level'] = train_data['sii'].apply(categorize_risk)


mild_patients = train_data[train_data['risk_level'] == 1]['id']
severe_patients = train_data[train_data['risk_level'] == 2]['id']


print("Mild SII Patient IDs:")
print(mild_patients.tolist())

print("\nSevere SII Patient IDs:")
print(severe_patients.tolist())

# Number of Patients with mild and severe SII

import pandas as pd
import matplotlib.pyplot as plt


train_data = pd.read_csv('train.csv')


def categorize_risk(sii):
    if sii < 1:
        return 0
    elif 1 <= sii < 2:
        return 1
    else:
        return 2


train_data['risk_level'] = train_data['sii'].apply(categorize_risk)


mild_patients_count = len(train_data[train_data['risk_level'] == 1])
severe_patients_count = len(train_data[train_data['risk_level'] == 2])


categories = ['Mild Risk', 'Severe Risk']
counts = [mild_patients_count, severe_patients_count]


plt.figure(figsize=(8, 5))
plt.bar(categories, counts, color=['lightgreen', 'salmon'])
plt.title('Number of Patients with Mild and Severe SII')
plt.xlabel('Risk Level')
plt.ylabel('Number of Patients')
plt.xticks(rotation=0)
plt.grid(axis='y')


plt.show()

# Number of Patients with mild and severe SII with ID's

import pandas as pd
import matplotlib.pyplot as plt


train_data = pd.read_csv('train.csv')


def categorize_risk(sii):
    if sii < 1:
        return 0
    elif 1 <= sii < 2:
        return 1
    else:
        return 2


train_data['risk_level'] = train_data['sii'].apply(categorize_risk)


mild_patients_ids = train_data[train_data['risk_level'] == 1]['id']
severe_patients_ids = train_data[train_data['risk_level'] == 2]['id']


mild_patients_count = mild_patients_ids.count()
severe_patients_count = severe_patients_ids.count()


categories = ['Mild Risk', 'Severe Risk']
counts = [mild_patients_count, severe_patients_count]


plt.figure(figsize=(8, 5))
plt.bar(categories, counts, color=['lightgreen', 'salmon'])
plt.title('Number of Patients with Mild and Severe SII')
plt.xlabel('Risk Level')
plt.ylabel('Number of Patients')
plt.xticks(rotation=0)
plt.grid(axis='y')

plt.show()


print("Mild SII Patient IDs:")
print(mild_patients_ids.tolist())

print("\nSevere SII Patient IDs:")
print(severe_patients_ids.tolist())

# Displays relevant factors such as **age, sex, CGAS score, BMI**, and **PCIAT score**, which contribute to the severity index (SII)

import pandas as pd


train_data = pd.read_csv('train.csv')


def categorize_risk(sii):
    if sii < 1:
        return 0
    elif 1 <= sii < 2:
        return 1
    else:
        return 2


train_data['risk_level'] = train_data['sii'].apply(categorize_risk)


mild_patients = train_data[train_data['risk_level'] == 1]
severe_patients = train_data[train_data['risk_level'] == 2]


print("Mild SII Patient IDs and Relevant Factors:")
print(mild_patients[['id', 'Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-BMI',
                     'Fitness_Endurance-Max_Stage', 'PAQ_A-PAQ_A_Total',
                     'PCIAT-PCIAT_Total']].to_string(index=False))


print("\nSevere SII Patient IDs and Relevant Factors:")
print(severe_patients[['id', 'Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-BMI',
                       'Fitness_Endurance-Max_Stage', 'PAQ_A-PAQ_A_Total',
                       'PCIAT-PCIAT_Total']].to_string(index=False))

# Classifying mild and severe patients from age **10-22**

import pandas as pd
import matplotlib.pyplot as plt


train_data = pd.read_csv('train.csv')


def categorize_risk(sii):
    if sii < 1:
        return 0
    elif 1 <= sii < 2:
        return 1
    else:
        return 2


train_data['risk_level'] = train_data['sii'].apply(categorize_risk)


mild_patients = train_data[train_data['risk_level'] == 1]
severe_patients = train_data[train_data['risk_level'] == 2]


mild_factors_count = {
    'Age': mild_patients['Basic_Demos-Age'].value_counts(),
    'BMI': mild_patients['Physical-BMI'].value_counts(),
    'Fitness Max Stage': mild_patients['Fitness_Endurance-Max_Stage'].value_counts(),
    'PAQ Total': mild_patients['PAQ_A-PAQ_A_Total'].value_counts(),
    'PCIAT Total': mild_patients['PCIAT-PCIAT_Total'].value_counts()
}

severe_factors_count = {
    'Age': severe_patients['Basic_Demos-Age'].value_counts(),
    'BMI': severe_patients['Physical-BMI'].value_counts(),
    'Fitness Max Stage': severe_patients['Fitness_Endurance-Max_Stage'].value_counts(),
    'PAQ Total': severe_patients['PAQ_A-PAQ_A_Total'].value_counts(),
    'PCIAT Total': severe_patients['PCIAT-PCIAT_Total'].value_counts()
}


plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
mild_factors_count['Age'].plot(kind='bar', color='lightgreen', alpha=0.7)
plt.title('Mild Risk Patients - Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.xticks(rotation=45)


plt.subplot(1, 2, 2)
severe_factors_count['Age'].plot(kind='bar', color='salmon', alpha=0.7)
plt.title('Severe Risk Patients - Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Factors contributing to SII for Patient ID 00008ff9 who has severe SII aged 5 years old

import pandas as pd


train_data = pd.read_csv('train.csv')


def categorize_risk(sii):
    if sii < 1:
        return 0
    elif 1 <= sii < 2:
        return 1
    else:
        return 2


train_data['risk_level'] = train_data['sii'].apply(categorize_risk)


patient_id = '00008ff9'
patient_data = train_data[train_data['id'] == patient_id]


if not patient_data.empty:

    relevant_factors = patient_data[['id', 'Basic_Demos-Age', 'Basic_Demos-Sex',
                                      'CGAS-CGAS_Score', 'Physical-BMI',
                                      'PCIAT-PCIAT_Total']]

    print(f"Factors contributing to SII for Patient ID {patient_id}:")
    print(relevant_factors.to_string(index=False))
else:
    print(f"No data found for Patient ID {patient_id}.")

# Factors contributing to SII for Patient ID 02c4cf7f with mild SII of age 9 years old

import pandas as pd


train_data = pd.read_csv('train.csv')


def categorize_risk(sii):
    if sii < 1:
        return 0
    elif 1 <= sii < 2:
        return 1
    else:
        return 2


train_data['risk_level'] = train_data['sii'].apply(categorize_risk)


patient_id = '02c4cf7f'
patient_data = train_data[train_data['id'] == patient_id]


if not patient_data.empty:

    relevant_factors = patient_data[['id', 'Basic_Demos-Age', 'Basic_Demos-Sex',
                                      'CGAS-CGAS_Score', 'Physical-BMI',
                                      'PCIAT-PCIAT_Total']]

    print(f"Factors contributing to SII for Patient ID {patient_id}:")
    print(relevant_factors.to_string(index=False))
else:
    print(f"No data found for Patient ID {patient_id}.")

# Patient IDs with Severe SII and Age of 5

import pandas as pd


train_data = pd.read_csv('train.csv')


def categorize_risk(sii):
    if sii < 1:
        return 0
    elif 1 <= sii < 2:
        return 1
    else:
        return 2


train_data['risk_level'] = train_data['sii'].apply(categorize_risk)


severe_age_5_patients = train_data[(train_data['risk_level'] == 2) & (train_data['Basic_Demos-Age'] == 5)]


if not severe_age_5_patients.empty:

    print("Patient IDs with Severe SII and Age of 5:")
    print(severe_age_5_patients['id'].tolist())
else:
    print("No patients found with Severe SII and Age of 5.")

# Patient IDs with Severe SII and Age of :22

import pandas as pd


train_data = pd.read_csv('train.csv')


def categorize_risk(sii):
    if sii < 1:
        return 0
    elif 1 <= sii < 2:
        return 1
    else:
        return 2


train_data['risk_level'] = train_data['sii'].apply(categorize_risk)


severe_age_5_patients = train_data[(train_data['risk_level'] == 2) & (train_data['Basic_Demos-Age'] == 22)]


if not severe_age_5_patients.empty:

    print("Patient IDs with Severe SII and Age of :22")
    print(severe_age_5_patients['id'].tolist())
else:
    print("No patients found with Severe SII and Age of 22.")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


print("Data types in training data:")
print(train_data.dtypes)
print("\nAvailable columns in training data:")
print(train_data.columns.tolist())
print("\nAvailable columns in test data:")
print(test_data.columns.tolist())


numerical_cols = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
train_data[numerical_cols] = train_data[numerical_cols].fillna(train_data[numerical_cols].mean())


categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    train_data[col].fillna(train_data[col].mode()[0], inplace=True)


print("Missing values in training data after imputation:")
print(train_data.isnull().sum())


def categorize_risk(sii):
    if sii < 1:
        return 0
    elif 1 <= sii < 2:
        return 1
    else:
        return 2


train_data['risk_level'] = train_data['sii'].apply(categorize_risk)


features = [
    'Basic_Demos-Age',
    'Basic_Demos-Sex',
    'Physical-BMI',
    'Fitness_Endurance-Max_Stage',
    'FGC-FGC_CU',
    'BIA-BIA_BMI',
    'PAQ_A-PAQ_A_Total'
]


available_features = [feature for feature in features if feature in train_data.columns]
print("\nAvailable features for training:")
print(available_features)


X = train_data[available_features]
y = train_data['risk_level'].values


y = to_categorical(y, num_classes=3)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numerical_cols = [col for col in available_features if col not in categorical_cols]
categorical_cols = [col for col in categorical_cols if col in available_features]


numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numerical_cols),
                  ('cat', categorical_transformer, categorical_cols)],
    remainder='drop'
)


X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


print(f"Shape of processed training data: {X_train_processed.shape}")


model = keras.Sequential([
    layers.Input(shape=(X_train_processed.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(X_train_processed, y_train, epochs=50, batch_size=32, validation_data=(X_test_processed, y_test))


accuracy = model.evaluate(X_test_processed, y_test)
print(f'Test accuracy: {accuracy[1] * 100:.2f}%')


age_range_data = train_data[(train_data['Basic_Demos-Age'] >= 5) & (train_data['Basic_Demos-Age'] <= 12)]


age_range_X = age_range_data[available_features]


age_range_X_processed = preprocessor.transform(age_range_X)


predictions = model.predict(age_range_X_processed)


predicted_classes = np.argmax(predictions, axis=1)


class_labels = {0: 'Low', 1: 'Mild', 2: 'Severe'}
predicted_labels = [class_labels[i] for i in predicted_classes]


for i in range(len(age_range_data)):
    age = age_range_data['Basic_Demos-Age'].iloc[i]
    bmi = age_range_data['Physical-BMI'].iloc[i]
    prediction = predicted_labels[i]
    print(f"Age: {age}, BMI: {bmi} -> Predicted Risk: {prediction}")

import pandas as pd

# Load the dataset
train_data = pd.read_csv('train.csv')

# Function to categorize risk levels based on SII
def categorize_risk(sii):
    if sii < 1:
        return 0  # None
    elif 1 <= sii < 2:
        return 1  # Mild
    else:
        return 2  # Severe

# Apply the function to create a new column for risk level
train_data['risk_level'] = train_data['sii'].apply(categorize_risk)

# Extract patient IDs based on risk levels
mild_patients = train_data[train_data['risk_level'] == 1]['id']
severe_patients = train_data[train_data['risk_level'] == 2]['id']

# Create DataFrames for mild and severe patients
mild_patients_df = pd.DataFrame(mild_patients.tolist(), columns=['Mild SII Patient IDs'])
severe_patients_df = pd.DataFrame(severe_patients.tolist(), columns=['Severe SII Patient IDs'])

# Display the results in table format
print("Mild SII Patient IDs:")
print(mild_patients_df)

print("\nSevere SII Patient IDs:")
print(severe_patients_df)



import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
train_data = pd.read_csv('train.csv')

# Function to categorize risk levels based on SII
def categorize_risk(sii):
    if sii < 1:
        return 0  # None
    elif 1 <= sii < 2:
        return 1  # Mild
    else:
        return 2  # Severe

# Apply the function to create a new column for risk level
train_data['risk_level'] = train_data['sii'].apply(categorize_risk)

# Extract patient IDs based on risk levels
mild_patients_ids = train_data[train_data['risk_level'] == 1]['id']
severe_patients_ids = train_data[train_data['risk_level'] == 2]['id']

# Count of mild and severe patients
mild_patients_count = mild_patients_ids.count()
severe_patients_count = severe_patients_ids.count()

# Bar plot for mild and severe risk patients
categories = ['Mild Risk', 'Severe Risk']
counts = [mild_patients_count, severe_patients_count]

plt.figure(figsize=(8, 5))
plt.bar(categories, counts, color=['lightgreen', 'salmon'])
plt.title('Number of Patients with Mild and Severe SII')
plt.xlabel('Risk Level')
plt.ylabel('Number of Patients')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

# Create DataFrames for mild and severe patients
mild_patients_df = pd.DataFrame(mild_patients_ids.tolist(), columns=['Mild SII Patient IDs'])
severe_patients_df = pd.DataFrame(severe_patients_ids.tolist(), columns=['Severe SII Patient IDs'])

# Display the results in table format
print("Mild SII Patient IDs:")
print(mild_patients_df)

print("\nSevere SII Patient IDs:")
print(severe_patients_df)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical

# Load the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Check data types and available columns
print("Data types in training data:")
print(train_data.dtypes)
print("\nAvailable columns in training data:")
print(train_data.columns.tolist())
print("\nAvailable columns in test data:")
print(test_data.columns.tolist())

# Fill missing values for numerical columns with mean
numerical_cols = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
train_data[numerical_cols] = train_data[numerical_cols].fillna(train_data[numerical_cols].mean())

# Fill missing values for categorical columns with mode
categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    train_data[col].fillna(train_data[col].mode()[0], inplace=True)

# Check if there are still any missing values
print("Missing values in training data after imputation:")
print(train_data.isnull().sum())

# Create a new target variable based on 'sii'
def categorize_risk(sii):
    if sii < 1:
        return 0  # Low risk
    elif 1 <= sii < 2:
        return 1  # Mild risk
    else:
        return 2  # Severe risk

# Apply this function to the 'sii' column to create the 'risk_level' target variable
train_data['risk_level'] = train_data['sii'].apply(categorize_risk)

# Select features for training
features = [
    'Basic_Demos-Age',
    'Basic_Demos-Sex',
    'Physical-BMI',
    'Fitness_Endurance-Max_Stage',
    'FGC-FGC_CU',
    'BIA-BIA_BMI',
    'PAQ_A-PAQ_A_Total'
]

# Verify which features are available in the training data and filter them
available_features = [feature for feature in features if feature in train_data.columns]
print("\nAvailable features for training:")
print(available_features)

# Split features and target variable
X = train_data[available_features]
y = train_data['risk_level'].values  # Target variable is now the categorized 'risk_level'

# Convert the labels into one-hot encoding
y = to_categorical(y, num_classes=3)  # Three categories: 0, 1, 2 (Low, Mild, Severe)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical and numerical columns again for preprocessing based on available features
numerical_cols = [col for col in available_features if col not in categorical_cols]
categorical_cols = [col for col in categorical_cols if col in available_features]

# Create preprocessing pipelines for both numeric and categorical data
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numerical_cols),
                  ('cat', categorical_transformer, categorical_cols)],
    remainder='drop'  # Drop any remaining columns not specified in transformers
)

# Preprocess training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Print the shape of processed training data to debug
print(f"Shape of processed training data: {X_train_processed.shape}")

# Build the ANN model
model = keras.Sequential([
    layers.Input(shape=(X_train_processed.shape[1],)),  # Use the number of features from X_processed
    layers.Dense(64, activation='relu'),  # First hidden layer
    layers.Dense(32, activation='relu'),  # Second hidden layer
    layers.Dense(3, activation='softmax')  # Output layer with 3 neurons (for 3 classes)
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model and save history for plotting later
history = model.fit(X_train_processed, y_train, epochs=50, batch_size=32, validation_data=(X_test_processed, y_test))

# Evaluate the model on test set
accuracy = model.evaluate(X_test_processed, y_test)
print(f'Test accuracy: {accuracy[1] * 100:.2f}%')

# Plotting training history (accuracy and loss)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Make predictions for individuals with age 5-12 from original dataset (if needed)
age_range_data = train_data[(train_data['Basic_Demos-Age'] >= 13) & (train_data['Basic_Demos-Age'] <=22 )]

# Extract features for this specific age range individuals based on available features only
age_range_X = age_range_data[available_features]

# Normalize the data for these specific rows using the same preprocessor fitted on training data
age_range_X_processed = preprocessor.transform(age_range_X)

# Predict risk levels for these specific individuals
predictions = model.predict(age_range_X_processed)

# Convert predictions to class labels (0, 1 or 2)
predicted_classes = np.argmax(predictions, axis=1)

# Map predicted class indices to labels ('Low', 'Mild', 'Severe')
class_labels = {0: 'Low', 1: 'Mild', 2: 'Severe'}
predicted_labels = [class_labels[i] for i in predicted_classes]

# Show predictions for each individual in the age range
for i in range(len(age_range_data)):
    age = age_range_data['Basic_Demos-Age'].iloc[i]
    bmi = age_range_data['Physical-BMI'].iloc[i]
    prediction = predicted_labels[i]
    print(f"Age: {age}, BMI: {bmi} -> Predicted Risk: {prediction}")

# Plot a bar chart of predicted risk levels
risk_counts = {
    'Low': predicted_labels.count('Low'),
    'Mild': predicted_labels.count('Mild'),
    'Severe': predicted_labels.count('Severe')
}

# Plot the distribution of predicted risk levels
plt.figure(figsize=(8, 5))
plt.bar(risk_counts.keys(), risk_counts.values(), color=['green', 'yellow', 'red'])
plt.title('Distribution of Predicted Risk Levels (Age 13-22)')
plt.xlabel('Risk Level')
plt.ylabel('Count')
plt.show()
