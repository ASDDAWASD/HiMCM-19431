import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.DataFrame({
    'City': ['City A', 'City B', 'City C', 'City D', 'City E', 'City F', 'City G', 'City H'],
    'GDP_per_Capita': [42000, 85000, 68000, 55000, 82000, 36000, 52000, 48000],
    'Cost_of_Living': [82, 100, 88, 85, 91, 76, 74, 70],
    'Transit_Score': [98, 89, 92, 95, 94, 96, 78, 90],
    'Green_Space_Score': [22, 27, 33, 21, 47, 28, 30, 44],
    'Air_Quality_Index': [35, 42, 38, 45, 40, 58, 28, 3],
    'Crime_Rate': [380.7, 295.4, 512.1, 189.0, 420.5, 610.8, 245.3, 330.9],
    'Graduation_Rate': [88.5, 91.2, 89.4, 84.1, 86.7, 85.9, 87.3, 82.0]
})
metadata = pd.DataFrame({
    'GDP_per_Capita': {
        'Dimension': 'Economic Viability and Opportunity',
        'Direction': 'positive'
    },
    'Cost_of_Living': {
        'Dimension': 'Economic Viability and Opportunity',
        'Direction': 'positive'
    },
    'Transit_Score': {
        'Dimension': 'Quality of Life',
        'Direction': 'positive'
    },
    'Green_Space_Score': {
        'Dimension': 'Quality of Life',
        'Direction': 'positive'
    },
    'Air_Quality_Index': {
        'Dimension': 'Health and Safety',
        'Direction': 'negative'
    },
    'Crime_Rate': {
        'Dimension': 'Health and Safety',
        'Direction': 'negative'
    },
    'Graduation_Rate': {
        'Dimension': 'Intellectual and Educational Vitality',
        'Direction': 'positive'
    }
})

cities = data['City']
features = data.drop(columns=['City'])

negative_indicators = [
    indicator
    for indicator, attributes in metadata.items()
    if attributes['Direction'] == 'negative'
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
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

print('\nScaled features:')
print(scaled_features_df)

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

# drop indicators here

city_scores = {
    'Economic Viability and Opportunity': [],
    'Quality of Life': [],
    'Health and Safety': [],
    'Intellectual and Educational Vitality': []
}

for dimension in city_scores.keys():
    col_mask = metadata.loc['Dimension', scaled_features_df.columns] == dimension
    city_scores[dimension] = scaled_features_df.loc[:, col_mask].mean(axis=1).tolist()

city_scores_df = pd.DataFrame(city_scores)

print('\nCity scores:')
print(city_scores_df)

rankings = {
    'Economic Viability and Opportunity': 2,
    'Quality of Life': 3,
    'Health and Safety': 1,
    'Intellectual and Educational Vitality': 4
}

weights = np.array([
    sum(1/i for i in range(rank, len(rankings)+1)) / len(rankings)
    for rank in rankings.values()
])

criteria = list(rankings.keys())

city_scores_features = city_scores_df[criteria].values

norms = np.sqrt((city_scores_features**2).sum(axis=0))

R = city_scores_features / norms
V = R * weights

ideal_best = V.max(axis=0)
ideal_worst = V.min(axis=0)

D_best = np.sqrt(((V - ideal_best)**2).sum(axis=1))
D_worst = np.sqrt(((V - ideal_worst)**2).sum(axis=1))

topsis_scores = np.divide(D_worst, D_best + D_worst)

results = pd.DataFrame({
    'City': cities,
    'Distance to Best': D_best,
    'Distance to Worst': D_worst,
    'TOPSIS Score': topsis_scores
})

results = results.sort_values('TOPSIS Score', ascending=False).reset_index(drop=True)

print('\nResults')
print(results)
