from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns


train_features = pd.read_csv('./data/dengue_features_train.csv',
                             index_col=[0,1,2])

train_labels = pd.read_csv('./data/dengue_labels_train.csv',
                           index_col=[0,1,2])

                           
# Seperate data for San Juan
sj_train_features = train_features.loc['sj']
sj_train_labels = train_labels.loc['sj']

# Separate data for Iquitos
iq_train_features = train_features.loc['iq']
iq_train_labels = train_labels.loc['iq']

# Remove `week_start_date` string.
sj_train_features.drop('week_start_date', axis=1, inplace=True)
iq_train_features.drop('week_start_date', axis=1, inplace=True)

sj_train_features['total_cases'] = sj_train_labels.total_cases
iq_train_features['total_cases'] = iq_train_labels.total_cases

# compute the correlations
sj_correlations = sj_train_features.corr()
iq_correlations = iq_train_features.corr()
sj_correlations.to_csv('sj_cor.csv')
iq_correlations.to_csv('iq_cor.csv')


sj_correlations = sj_correlations[sj_correlations.columns[::-1]]
iq_correlations = iq_correlations[iq_correlations.columns[::-1]]
#draw heatmap

#draw sj heatmap
ax1 = sns.heatmap(sj_correlations)

for item in ax1.get_yticklabels():
    item.set_rotation(0)
for item in ax1.get_xticklabels():
    item.set_rotation(90)
plt.tight_layout()
plt.title('San Juan correlation')
plt.savefig('sj_cor.png',dpi = 100)
plt.clf()


ax = sns.heatmap(iq_correlations)
for item in ax.get_yticklabels():
    item.set_rotation(0)
for item in ax.get_xticklabels():
    item.set_rotation(90)
plt.tight_layout()
plt.title('Iquitos correlation')
plt.savefig('iq_cor.png',dpi = 100)
plt.clf()



