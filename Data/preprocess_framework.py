import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

data = pd.read_csv("Data/Dataset-CityIndicators.csv", skiprows=[0, 2])
data = data.drop(columns=['% pop 3 & older enrolled in k-12 school'])

dimensions = {
    'Economic Viability and Opportunity': data.columns[2:7],
    'Intellectual and Educational Vitality': data.columns[7:13],
    'Health and Safety': data.columns[13:18],
    'Quality of Life': data.columns[18:21],
}

cities = data['Geographic Area Name']
features = data.drop(columns=['Geographic Area Name', 'total population'])
features = features.apply(pd.to_numeric, errors='coerce')

negative_indicators = [
    'Gini index of income inequality',
    'housing burden',
    'rent burden',
    '% ppl below poverty lvl',
    'student/teacher ratio',
    'Mortality (deaths per 100,000)',
    'crime rate per 100,000 ppl',
    'mean commute time to work (min)',
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

imputer = SimpleImputer()
imputed_features = imputer.fit_transform(features)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(imputed_features)
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

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

abs_loadings = loadings_df.abs()

primary = abs_loadings.max(axis=1)
alternative = abs_loadings.apply(
    lambda x: x.nlargest(2).iloc[-1], axis=1
)

cross_loaded_indicators = loadings_df[(primary <= 0.40) | (alternative >= 0.30) | ((primary - alternative) < 0.20)]

cross_loadings = pd.DataFrame({
    'Primary': primary,
    'Alternative': alternative
})

print('\nLoadings of cross-loaded indicators:')
print(cross_loadings.round(3))

# drop indicators here

city_scores = {
    'Economic Viability and Opportunity': [],
    'Quality of Life': [],
    'Health and Safety': [],
    'Intellectual and Educational Vitality': []
}

for dim, indicators in dimensions.items():
    city_scores[dim] = scaled_features_df[indicators].mean(axis=1).tolist()

city_scores_df = pd.DataFrame(city_scores)
city_scores_df.insert(0, "City", cities)

print('\nCity scores:')
print(city_scores_df.round(3))

rankings = {
    'Economic Viability and Opportunity': 4,
    'Quality of Life': 3,
    'Health and Safety': 2,
    'Intellectual and Educational Vitality': 1
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

# plt.figure(figsize=(10, 6))
# sns.heatmap(corr_matrix, annot=True, fmt='.2f')
# plt.title('Correlation Matrix Heatmap')
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.show()
# 
# plt.figure(figsize=(10, 6))
# sns.heatmap(loadings_df, annot=True, fmt='.2f')
# plt.title('Loadings Matrix Heatmap')
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.show()
