import numpy as np
import pandas as pd 
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor as RF

np.random.seed(0)

feature = pd.read_csv('data/dengue_features_train.csv', infer_datetime_format=True)
label = pd.read_csv('data/dengue_labels_train.csv')

df = pd.merge(feature, label, how='outer', on=label.columns.tolist()[:-1])
df = df.join(pd.get_dummies(df.city))

ignore_feature_list = ['city', 'sj', 'iq', 'precipitation_amount_mm', 'reanalysis_avg_temp_k', 'week',
   'total_cases', 'week_start_date', 'reanalysis_specific_humidity_g_per_kg', 'station_diur_temp_rng_c']

target = 'total_cases'

test = pd.read_csv('data/dengue_features_test.csv')
test[target] = np.nan
test = test.join(pd.get_dummies(test.city))


def preprocess(df, training = True):
    df['cosweekofyear'] = np.cos((df['weekofyear']/52)*2*np.pi)
    df_sj = df.loc[df.city == 'sj']
    df_iq = df.loc[df.city == 'iq']
    if training:
       df_sj.loc[:, [target]] = df_sj.loc[:, target].shift(1)
       df_iq.loc[:, [target]] = df_iq.loc[:, target].shift(1)
    df_sj.fillna(method='ffill', inplace=True)
    df_iq.fillna(method='ffill', inplace=True)
    return df_sj, df_iq    

sj, iq = preprocess(df)
test_sj, test_iq = preprocess(test, False)
for i in range(5):
    test_sj[target + '_lag_' + str(i+1)] = np.nan 
    test_iq[target + '_lag_' + str(i+1)] = np.nan 

def add_labels(df, feature_list, n_lag=0):
    
    new_df = df.copy()
    
    for original_feature in feature_list:
        for n in range(n_lag):
            lagging_feature_name = original_feature+'_lag_'+ str(n+1)
            new_df.loc[:,lagging_feature_name] = new_df.loc[:,original_feature].shift(n+1)
    new_df = new_df.iloc[n_lag:,:]        
    return new_df

nlag_sj = 5 
nlag_iq = 5 
sj_lag = sj.copy()
iq_lag = iq.copy()
sj_lag = add_labels(sj, [target], nlag_sj)
iq_lag = add_labels(iq, [target], nlag_iq)

predictors = [feat for feat in sj_lag.columns.tolist() if feat not in ignore_feature_list ]

test_sj_lag = pd.concat([sj_lag.iloc[(sj_lag.shape[0]-nlag_sj):,: ], test_sj])
test_iq_lag = pd.concat([iq_lag.iloc[(iq_lag.shape[0]-nlag_iq):,: ], test_iq])

rf_sj = RF(n_estimators = 1500, max_features=10)
rf_iq = RF(n_estimators = 1500, max_features=10)
sj_lag.fillna(method='bfill', inplace=True)
iq_lag.fillna(method='bfill', inplace=True)
rf_sj.fit(sj_lag[predictors].values, sj_lag[target].values)
rf_iq.fit(iq_lag[predictors].values, iq_lag[target].values)


sj_pred = []
iq_pred = []

sj_column = test_sj_lag.columns
iq_column = test_iq_lag.columns
for i in range(nlag_sj, test_sj_lag.shape[0]):
    for j in range(nlag_sj):
        test_sj_lag.iloc[i, sj_column.get_loc(target+'_lag_'+str(j+1))] = test_sj_lag.iloc[i-j-1][target]
    pred = test_sj_lag.iloc[i:i+1][predictors].values
    ans = rf_sj.predict(pred)[0]
    sj_pred.append(ans)
    test_sj_lag.iloc[i, sj_column.get_loc(target)] = ans

for i in range(nlag_iq, test_iq_lag.shape[0]):
    for j in range(nlag_iq):
        test_iq_lag.iloc[i, iq_column.get_loc(target+'_lag_'+str(j+1))] = test_iq_lag.iloc[i-j-1][target]
    pred = test_iq_lag.iloc[i:i+1][predictors].values
    ans = rf_iq.predict(pred)[0]
    iq_pred.append(ans)
    test_iq_lag.iloc[i, iq_column.get_loc(target)] = ans

sj_pred = np.array(sj_pred)  
iq_pred = np.array(iq_pred)

result = np.concatenate([sj_pred, iq_pred], axis=0)


y_pred = np.round(result).astype(int)
result_df = test[['city', 'year', 'weekofyear']].copy()
result_df['total_cases'] = y_pred 
result_df.to_csv('res.csv', index=False)
