import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = {
    'City': [
        'City A',
        'City B',
        'City C',
        'City D',
        'City E',
        'City F',
        'City G',
        'City H',
    ],
    'GDP_per_Capita': [42000, 85000, 68000, 55000, 82000, 36000, 52000, 48000],
    'Cost_of_Living': [82, 100, 88, 85, 91, 76, 74, 70],
    'Transit_Score': [98, 89, 92, 95, 94, 96, 78, 90],
    'Green_Space_Score': [22, 27, 33, 21, 47, 28, 30, 44],
    'Air_Quality_Index': [35, 42, 38, 45, 40, 58, 28, 3],
    'Crime_Rate': [380.7, 295.4, 512.1, 189.0, 420.5, 610.8, 245.3, 330.9]
}
df = pd.DataFrame(data)

cities = df['City']
features = df.drop(columns=['City'])

negative_indicators = [
    'Air_Quality_Index',
    'Crime_Rate'
]
for indicator in negative_indicators:
    features[indicator] = features[indicator] * -1

corr_matrix = features.corr()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
pairs = upper_tri.stack().reset_index()
pairs.columns = ['Indicator_1', 'Indicator_2', 'Correlation']
high_corr = pairs[pairs['Correlation'].abs() > 0.7]

print('Pairs with correlation >0.7:')
print(high_corr.round(3))

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

pca = PCA()
principal_components = pca.fit_transform(scaled_features)

pc_columns = [f'PC{i}' for i in range(principal_components.shape[1])]
pca_df = pd.DataFrame(data=principal_components, columns=pc_columns)
pca_df.insert(0, 'City', cities)

print('\nIndicators in PCA space:')
print(pca_df.round(3))

explained_variance = pca.explained_variance_ratio_
for i, var in enumerate(explained_variance):
    print(f'PC{i} explains {var:.2%} of the variance')

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

loadings_df = pd.DataFrame(loadings, columns=pc_columns, index=features.columns)

print('\nLoadings:')
print(loadings_df.round(3))

cross_loaded_indicators = loadings_df[(loadings_df.abs() > 0.40).sum(axis=1) > 1]
print('\nLoadings of cross-loaded indicators:')
print(cross_loaded_indicators.where(cross_loaded_indicators.abs() > 0.4).round(3))
