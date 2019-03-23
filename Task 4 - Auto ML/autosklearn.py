!pip install auto-sklearn

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.regression

loandata_df = pd.read_csv('Data/loan_cleaned.csv', index_col='id')

#see the columns in our data
loandata_df.info()

# take a look at the head of the dataset
loandata_df.head()


#create our X and y
X = loandata_df.drop('int_rate', axis=1)
y = loandata_df['int_rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)


automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_regression_example_tmp',
        output_folder='/tmp/autosklearn_regression_example_out',
    )
    automl.fit(X_train, y_train, dataset_name='loandata_df',
               feat_type=feature_types)

print(automl.show_models())
predictions = automl.predict(X_test)
print("R2 score:", sklearn.metrics.r2_score(y_test, predictions))


