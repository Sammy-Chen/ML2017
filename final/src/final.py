import numpy as np
import pandas as pd 
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from sklearn import metrics

np.random.seed(0)

feature = pd.read_csv('mydata/data/dengue_features_train.csv', infer_datetime_format=True)
label = pd.read_csv('mydata/data/dengue_labels_train.csv')

df = pd.merge(feature, label, how='outer', on=label.columns.tolist()[:-1])
df = df.join(pd.get_dummies(df.city))

ignore_feature_list = ['city', 'sj', 'iq', 'precipitation_amount_mm', 'reanalysis_avg_temp_k', 'week',
   'total_cases', 'week_start_date', 'reanalysis_specific_humidity_g_per_kg', 'station_diur_temp_rng_c']
target = 'total_cases'

test = pd.read_csv('mydata/data/dengue_features_test.csv')
test[target] = np.nan
test = test.join(pd.get_dummies(test.city))


def processing_function(df, training = True):

    df['cosweekofyear'] = np.cos((df['weekofyear']/52)*2*np.pi)
    
    df_sj = df.loc[df.city == 'sj']
    df_iq = df.loc[df.city == 'iq']

    if training:
       df_sj.loc[:, [target]] = df_sj.loc[:, target].shift(1)
       df_iq.loc[:, [target]] = df_iq.loc[:, target].shift(1)
    
    df_sj.fillna(method='ffill', inplace=True)
    df_iq.fillna(method='ffill', inplace=True)
    
    return df_sj, df_iq    

sj, iq = processing_function(df)
test_sj, test_iq = processing_function(test, False)

for i in range(5):
    test_sj[target + '_lag_' + str(i+1)] = np.nan 
    test_iq[target + '_lag_' + str(i+1)] = np.nan 

lagging_feature_list = [target]

def add_lagging_feature(df, lagging_feature_list, n_lag=0):
    
    new_df = df.copy()
    for original_feature in lagging_feature_list:
        for n in range(n_lag):
            lagging_feature_name = original_feature+'_lag_'+ str(n+1)
            new_df.loc[:,lagging_feature_name] = new_df.loc[:,original_feature].shift(n+1)
    new_df = new_df.iloc[n_lag:,:]        
    return new_df

nlag_sj = 5 
nlag_iq = 5 
sj_lag = sj.copy()
iq_lag = iq.copy()
sj_lag = add_lagging_feature(sj, lagging_feature_list, nlag_sj)
iq_lag = add_lagging_feature(iq, lagging_feature_list, nlag_iq)

lagging_predictors_sj = [feat for feat in sj_lag.columns.tolist() if feat not in ignore_feature_list ]
lagging_predictors_iq = [feat for feat in iq_lag.columns.tolist() if feat not in ignore_feature_list ]

test_sj_lag = pd.concat([sj_lag.iloc[(sj_lag.shape[0]-nlag_sj):,: ], test_sj])
test_iq_lag = pd.concat([iq_lag.iloc[(iq_lag.shape[0]-nlag_iq):,: ], test_iq])

h2o.init()
h2o.remove_all() 


sj_frame = h2o.H2OFrame(python_obj=sj_lag.to_dict('list'))

iq_frame = h2o.H2OFrame(python_obj=iq_lag.to_dict('list'))

rf_sj = H2ORandomForestEstimator(
    model_id="random_forest_sj",
    ntrees=1500,
    mtries=10,
    seed=0)

rf_sj.train(lagging_predictors_sj, target, training_frame=sj_frame,validation_frame=None)

rf_iq = H2ORandomForestEstimator(
    model_id="random_forest_iq",
    ntrees=1500,
    mtries=10,
    seed=0)

rf_iq.train(lagging_predictors_iq, target, training_frame=iq_frame,validation_frame=None)

sj_pred = []
iq_pred = []

sj_column = test_sj_lag.columns
iq_column = test_iq_lag.columns
for i in range(nlag_sj, test_sj_lag.shape[0]):
    for j in range(nlag_sj):
        test_sj_lag.iloc[i, sj_column.get_loc(target+'_lag_'+str(j+1))] = test_sj_lag.iloc[i-j-1][target]
    pred_frame = h2o.H2OFrame(python_obj=test_sj_lag.iloc[i:i+1][lagging_predictors_sj].to_dict('list'))
    ans = rf_sj.predict(pred_frame).as_data_frame(use_pandas=True)['predict'].values[0]
    sj_pred.append(ans)
    test_sj_lag.iloc[i, sj_column.get_loc(target)] = ans

for i in range(nlag_iq, test_iq_lag.shape[0]):
    for j in range(nlag_iq):
        test_iq_lag.iloc[i, iq_column.get_loc(target+'_lag_'+str(j+1))] = test_iq_lag.iloc[i-j-1][target]
    pred_frame = h2o.H2OFrame(python_obj=test_iq_lag.iloc[i:i+1][lagging_predictors_iq].to_dict('list'))
    ans = rf_iq.predict(pred_frame).as_data_frame(use_pandas=True)['predict'].values[0]
    iq_pred.append(ans)
    test_iq_lag.iloc[i, iq_column.get_loc(target)] = ans

sj_pred = np.array(sj_pred)  
iq_pred = np.array(iq_pred)

result = np.concatenate([sj_pred, iq_pred], axis=0)


y_pred = np.round(result).astype(int)
print(y_pred)
print(y_pred.mean())
result_df = test[['city', 'year', 'weekofyear']].copy()
result_df['total_cases'] = y_pred 

result_df.to_csv('res.csv', index=False)
