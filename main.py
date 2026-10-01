import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 110

df = pd.read_csv('housing.csv')

RU_NAMES = {
    'longitude': 'Долгота',
    'latitude': 'Широта',
    'housing_median_age': 'Медианный возраст жилья',
    'total_rooms': 'Всего комнат',
    'total_bedrooms': 'Всего спален',
    'population': 'Население',
    'households': 'Домохозяйства',
    'median_income': 'Медианный доход',
    'median_house_value': 'Медианная стоимость дома',
    'ocean_proximity': 'Близость к океану'
}

df_ru = df.rename(columns=RU_NAMES)

print("=" * 80)
print(f"Размерность датасета: {df_ru.shape[0]} строк, {df_ru.shape[1]} признаков")
print("=" * 80)
print("\nПервые 5 строк датасета:")
print(df_ru.head().to_string())

print("\nПроверка пропущенных значений:")
print(df_ru.isnull().sum().to_string())

duplicates_count = df_ru.duplicated().sum()
print(f"\nКоличество повторяющихся строк (дубликатов): {duplicates_count}")

df_ru['Всего спален'] = df_ru['Всего спален'].fillna(df_ru['Всего спален'].median())

num_cols = df_ru.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_ru.select_dtypes(include=['object', 'category']).columns.tolist()

stats_df = df_ru[num_cols].describe().T
stats_df['Медиана'] = df_ru[num_cols].median()
stats_df['Мода'] = [df_ru[col].mode()[0] for col in num_cols]
stats_df['IQR'] = stats_df['75%'] - stats_df['25%']
stats_df['Асимметрия'] = df_ru[num_cols].skew()

stats_df = stats_df.rename(columns={
    'count': 'Количество',
    'mean': 'Среднее',
    'std': 'Станд. откл.',
    'min': 'Минимум',
    'max': 'Максимум'
})

cols_order = ['Среднее', 'Станд. откл.', 'Медиана', 'Мода', 'IQR', 'Минимум', 'Максимум', 'Асимметрия']
print("=" * 95)
print("ОПИСАТЕЛЬНАЯ СТАТИСТИКА ЧИСЛОВЫХ ПЕРЕМЕННЫХ:")
print("=" * 95)
print(stats_df[cols_order].round(2).to_string())

if len(cat_cols) > 0:
    print("\n" + "=" * 95)
    print("ЧАСТОТНЫЙ АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ:")
    print("=" * 95)
    for col in cat_cols:
        counts = df_ru[col].value_counts()
        percentages = df_ru[col].value_counts(normalize=True) * 100
        cat_summary = pd.DataFrame({'Абсолютная частота': counts, 'Относительная доля (%)': percentages.round(2)})
        print(f"\nПризнак: {col}")
        print(cat_summary.to_string())

fig, axes = plt.subplots(len(num_cols), 2, figsize=(12, 3 * len(num_cols)))
for i, col in enumerate(num_cols):
    sns.histplot(df_ru[col], kde=True, ax=axes[i, 0], color='skyblue')
    axes[i, 0].set_title(f'Гистограмма: {col}')
    sns.boxplot(x=df_ru[col], ax=axes[i, 1], color='lightgreen')
    axes[i, 1].set_title(f'Коробчатая диаграмма: {col}')
plt.tight_layout()
plt.show()

df_encoded = pd.get_dummies(df_ru, columns=cat_cols, drop_first=True, dtype=int)

target_col = 'Медианная стоимость дома'
X = df_encoded.drop(columns=[target_col])
y = df_encoded[target_col]

plt.figure(figsize=(10, 8))
corr_matrix = X.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Матрица корреляций признаков')
plt.show()

X_vif = sm.add_constant(X).astype(float)
vif_data = pd.DataFrame()
vif_data["Признак"] = X_vif.columns
vif_values = []
for i in range(X_vif.shape[1]):
    try:
        val = variance_inflation_factor(X_vif.values, i)
    except Exception:
        val = np.nan
    vif_values.append(val)
vif_data["VIF"] = vif_values

print("=" * 80)
print("FACTOR INFLATION DISPERSION (VIF):")
print("=" * 80)
print(vif_data.round(2).to_string(index=False))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Линейная регрессия': LinearRegression(),
    'Ridge (Гребневая)': Ridge(alpha=10.0),
    'Lasso (Лассо)': Lasso(alpha=1.0, max_iter=10000)
}

def evaluate_models(models_dict, X_tr, X_te, y_tr, y_te):
    results = []
    for name, model in models_dict.items():
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=kf, scoring='neg_root_mean_squared_error')
        cv_rmse = -cv_scores.mean()
        
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        r2 = r2_score(y_te, preds)
        mape = mean_absolute_percentage_error(y_te, preds) * 100
        
        results.append({
            'Модель': name,
            'CV RMSE': round(cv_rmse, 2),
            'Test RMSE': round(rmse, 2),
            'Test R²': round(r2, 4),
            'Test MAPE (%)': round(mape, 2)
        })
    return pd.DataFrame(results)

results_before_pca = evaluate_models(models, X_train_scaled, X_test_scaled, y_train, y_test)
print("=" * 80)
print("МЕТРИКИ КАЧЕСТВА МОДЕЛЕЙ ДО ПРИМЕНЕНИЯ PCA:")
print("=" * 80)
print(results_before_pca.to_string(index=False))

pca_full = PCA()
pca_full.fit(X_train_scaled)

exp_var_ratio = pca_full.explained_variance_ratio_
cum_exp_var = np.cumsum(exp_var_ratio)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(exp_var_ratio) + 1), exp_var_ratio, 'ro-', label='Доля объясненной дисперсии')
plt.plot(range(1, len(cum_exp_var) + 1), cum_exp_var, 'bs--', label='Кумулятивная дисперсия')
plt.axhline(y=0.85, color='g', linestyle=':', label='Порог 85%')
plt.xlabel('Количество главных компонент')
plt.ylabel('Доля дисперсии')
plt.title('График каменистой осыпи (Scree Plot)')
plt.legend(loc='best')
plt.show()

n_components = np.argmax(cum_exp_var >= 0.85) + 1
print(f"Количество компонент для объяснения >= 85% дисперсии: {n_components}")

pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components)], index=X.columns)
print("\nФАКТОРНЫЕ НАГРУЗКИ ГЛАВНЫХ КОМПОНЕНТ:")
print(loadings.round(3).to_string())

results_after_pca = evaluate_models(models, X_train_pca, X_test_pca, y_train, y_test)
print("=" * 80)
print("МЕТРИКИ КАЧЕСТВА МОДЕЛЕЙ ПОСЛЕ PCA:")
print("=" * 80)
print(results_after_pca.to_string(index=False))

comparison_df = pd.merge(results_before_pca, results_after_pca, on='Модель', suffixes=(' (До PCA)', ' (После PCA)'))
print("\n" + "=" * 95)
print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТРИК КАЧЕСТВА:")
print("=" * 95)
print(comparison_df.to_string(index=False))

plt.figure(figsize=(10, 5))
x_labels = comparison_df['Модель']
x_axis = np.arange(len(x_labels))
plt.bar(x_axis - 0.2, comparison_df['Test RMSE (До PCA)'], 0.4, label='RMSE (До PCA)', color='cornflowerblue')
plt.bar(x_axis + 0.2, comparison_df['Test RMSE (После PCA)'], 0.4, label='RMSE (После PCA)', color='coral')
plt.xticks(x_axis, x_labels)
plt.ylabel('RMSE')
plt.title('Сравнение RMSE моделей до и после применения PCA')
plt.legend()
plt.show()