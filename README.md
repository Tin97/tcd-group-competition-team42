# tcd-group-competition-team42
Repository for the Income prediction group competition

We first replaced all invalid values in each column and used target labeling to replace all string values with numeric values.
The next step was dropping certain columns that did not affect the result (Hair Color, Wears Glasses, Instance..)
After that we created the regressor (CatBoostRegressor) and trained our model.
Finally we made predictions and calculated the mean absolute error and wrote our results to a csv file. 
