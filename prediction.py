import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import neural_network

data_train = pd.read_csv('train2.csv')
data_test = pd.read_csv('test2.csv')

numeric_features = ['Year of Record', 'Age','Crime Level in the City of Employement','Satisfation with employer', 'Gender', 'University Degree', 'Country', 'Size of City','Profession',]
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

#categorical_features = ['Satisfation with employer', 'Gender', 'University Degree']
#categorical_transformer = Pipeline(steps=[
#    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
        ])

#Create a pipeline and use a preprocessor from above and CatBoostRegressor model
regressor = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', CatBoostRegressor(iterations=2000, depth=10, learning_rate=0.01, loss_function='MAE'))])

#execute following using a gpu
with tf.device('gpu'):
    #replace the string with the mapped mean value
    meansCountry = data_train.groupby('Country')['Total Yearly Income [EUR]'].mean()
    data_train['Country'] = data_train['Country'].map(meansCountry)
    data_test['Country'] = data_test['Country'].map(meansCountry)

    #replace the string with the mapped mean value
    meansProfession = data_train.groupby('Profession')['Total Yearly Income [EUR]'].mean()
    data_train['Profession'] = data_train['Profession'].map(meansProfession)
    data_test['Profession'] = data_test['Profession'].map(meansProfession)

    #replace the string with the mapped mean value
    meansSatisfaction = data_train.groupby('Satisfation with employer')['Total Yearly Income [EUR]'].mean()
    data_train['Satisfation with employer'] = data_train['Satisfation with employer'].map(meansSatisfaction)
    data_test['Satisfation with employer'] = data_test['Satisfation with employer'].map(meansSatisfaction)

    data_train["Gender"] = data_train["Gender"].replace('f','female')
    data_train["Gender"] = data_train["Gender"].replace('0','unknown')
    data_test["Gender"] = data_test["Gender"].replace('f','female')
    data_test["Gender"] = data_test["Gender"].replace('0','unknown')

    #replace the string with the mapped mean value
    meansGender = data_train.groupby('Gender')['Total Yearly Income [EUR]'].mean()
    data_train['Gender'] = data_train['Gender'].map(meansGender)
    data_test['Gender'] = data_test['Gender'].map(meansGender)

    data_test['University Degree'] = data_test['University Degree'].replace('0','No')
    data_train['University Degree'] = data_train['University Degree'].replace('0','No')

    #replace the string with the mapped mean value
    meansDegree = data_train.groupby('University Degree')['Total Yearly Income [EUR]'].mean()
    data_train['University Degree'] = data_train['University Degree'].map(meansDegree)
    data_test['University Degree'] = data_test['University Degree'].map(meansDegree)

    data_train['Housing Situation'] = data_train['Housing Situation'].replace('nA',0)
    data_train['Housing Situation'] = data_train['Housing Situation'].astype(str)
    data_test['Housing Situation'] = data_test['Housing Situation'].replace('nA',0)
    data_test['Housing Situation'] = data_test['Housing Situation'].astype(str)

    meansHouse = data_train.groupby('Housing Situation')['Total Yearly Income [EUR]'].mean()
    data_train['Housing Situation'] = data_train['Housing Situation'].map(meansHouse)
    data_test['Housing Situation'] = data_test['Housing Situation'].map(meansHouse)

    data_train['Yearly Income in addition to Salary (e.g. Rental Income)'] = data_train['Yearly Income in addition to Salary (e.g. Rental Income)'].str.split(" ", n = 1, expand = True)[0]
    data_train['Yearly Income in addition to Salary (e.g. Rental Income)'] = data_train['Yearly Income in addition to Salary (e.g. Rental Income)'].astype(float)

    data_test['Yearly Income in addition to Salary (e.g. Rental Income)'] = data_test['Yearly Income in addition to Salary (e.g. Rental Income)'].str.split(" ", n = 1, expand = True)[0]
    data_test['Yearly Income in addition to Salary (e.g. Rental Income)'] = data_test['Yearly Income in addition to Salary (e.g. Rental Income)'].astype(float)

    data_train['Crime Level in the City of Employement'] = pd.to_numeric(data_train['Crime Level in the City of Employement'], errors='coerce').fillna(data_train['Crime Level in the City of Employement'].mean())
    data_test['Crime Level in the City of Employement'] = pd.to_numeric(data_test['Crime Level in the City of Employement'], errors='coerce').fillna(data_test['Crime Level in the City of Employement'].mean())

    data_train['Work Experience in Current Job [years]'] = data_train['Work Experience in Current Job [years]'].replace('#NUM!',0)
    data_test['Work Experience in Current Job [years]'] = data_test['Work Experience in Current Job [years]'].replace('#NUM!',0)

    data_test['Work Experience in Current Job [years]'] = pd.to_numeric(data_test['Work Experience in Current Job [years]'])
    data_train['Work Experience in Current Job [years]'] = pd.to_numeric(data_train['Work Experience in Current Job [years]'])


print("done")
X = data_train.drop('Instance', axis=1)
#X = X.drop('Housing Situation', axis=1)
#X = X.drop('Crime Level in the City of Employement', axis=1)
#X = X.drop('Yearly Income in addition to Salary (e.g. Rental Income)', axis=1)
X = X.drop('Hair Color', axis=1)
X = X.drop('Wears Glasses', axis=1)
X = X.drop('Total Yearly Income [EUR]', axis=1)
y = data_train['Total Yearly Income [EUR]']

X_train = data_train.drop('Instance', axis=1)
#X_train = X_train.drop('Crime Level in the City of Employement', axis=1)
#X_train = X_train.drop('Work Experience in Current Job [years]', axis=1)
#X_train = X_train.drop('Yearly Income in addition to Salary (e.g. Rental Income)', axis=1)
X_train = X_train.drop('Hair Color', axis=1)
#X_train = X_train.drop('Housing Situation', axis=1)
X_train = X_train.drop('Wears Glasses', axis=1)
X_train = X_train.drop('Total Yearly Income [EUR]', axis=1)
y_train = data_train['Total Yearly Income [EUR]']

X_test = data_test.drop('Instance', axis=1)
#X_test = X_test.drop('Crime Level in the City of Employement', axis=1)
#X_test = X_test.drop('Work Experience in Current Job [years]', axis=1)
#X_test = X_test.drop('Yearly Income in addition to Salary (e.g. Rental Income)', axis=1)
#X_test = X_test.drop('Housing Situation', axis=1)
X_test = X_test.drop('Hair Color', axis=1)
X_test = X_test.drop('Wears Glasses', axis=1)
instance = data_test['Instance']
X_test = X_test.drop('Total Yearly Income [EUR]', axis=1)
y_test = data_test['Total Yearly Income [EUR]']
y_addition = data_test['Yearly Income in addition to Salary (e.g. Rental Income)']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

#execute following using a gpu
with tf.device('gpu'):
    regressor = CatBoostRegressor(iterations=2000, depth=10, learning_rate=0.01, loss_function='MAE')
    regressor.fit(X_train, y_train,cat_features=None,eval_set=(X_valid, y_valid))
    y_pred = regressor.predict(X_valid)
    #y_pred = y_pred + y_addition
    print(mean_absolute_error(y_valid, y_pred))

    y_pred = regressor.predict(X_test)

    df = pd.DataFrame({'Instance': instance, 'Total Yearly Income [EUR]': y_pred})
    df.to_csv('submission.csv', sep= ',', index = False)

print("done")

#print(np.sqrt(mean_squared_error(y_test, y_pred)))
