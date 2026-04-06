from utils import db_connect
engine = db_connect()
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


# your code here


#Carga y limpieza de datos.
# Se carga la base de datos a trabajar desde data-raw, data set de Airbnb en Nueva York, 2019.

df = pd.read_csv('/workspaces/Proyecto-de-Preprocesamiento-de-Datos/data/raw/AB_NYC_2019.csv') 

print(df.head())    

total_data = df

print(total_data.info())
print(total_data.shape)
print (total_data.describe())

print(total_data.isnull().sum())
# Se observan las características de la base de datos, se observa que hay 48895 filas y 16 columnas, con algunas columnas con valores nulos.
# Se observa que hay 5 columnas con valores nulos, siendo 'name' y 'host_name' las que más valores nulos tienen,
# con 21 y 16 respectivamente. Las otras columnas con valores nulos son 'reviews_per_month', 'last_review' y 'neighbourhood_group',
# con 10052, 10052 y 0 valores nulos respectivamente.

# Se eliminan duplicados.
print(f"The number of duplicated Name records is: {total_data['name'].duplicated().sum()}")
print(f"The number of duplicated Host ID records is: {total_data['host_id'].duplicated().sum()}")
print(f"The number of duplicated ID records is: {total_data['id'].duplicated().sum()}")

# Se elimininan datos irrlevantes para prediccion de analisis en precios o creacion de modelos de machine learning,
# como 'id', 'name', 'host_name', 'last_review' y 'reviews_per_month'.
total_data = total_data.drop(['id', 'name', 'host_name', 'last_review', 'reviews_per_month'], axis=1)
total_data.info()
print(total_data.head())

#-------------------------------------------------------------------------------------------------------------------------
# Análisis de variables univariantes
#Categoricas
fig, axis = plt.subplots(2, 3, figsize=(15, 10))  

# Variables categóricas: usar countplot en lugar de histplot
sns.countplot(ax=axis[0,0], data=total_data, x="neighbourhood_group")
axis[0,0].set_title("Distribución por Grupo de Vecindario")
axis[0,0].tick_params(axis='x', rotation=45)  # Rotar etiquetas si es necesario

sns.countplot(ax=axis[0,1], data=total_data, x="neighbourhood")
axis[0,1].set_title("Distribución por Vecindario")
axis[0,1].tick_params(axis='x', rotation=90)  # Rotar para evitar superposición
axis[0,1].set_xticks([])  # Ocultar ticks si hay muchos

sns.countplot(ax=axis[0,2], data=total_data, x="room_type")
axis[0,2].set_title("Distribución por Tipo de Habitación")

# Variables numéricas: usar histplot
sns.histplot(ax=axis[1,0], data=total_data, x="host_id", bins=50)
axis[1,0].set_title("Distribución de Host ID")

sns.histplot(ax=axis[1,1], data=total_data, x="availability_365", bins=50)
axis[1,1].set_title("Distribución de Disponibilidad 365")

# Eliminar el último subplot vacío
fig.delaxes(axis[1, 2])

# Ajustar el diseño
plt.tight_layout()
plt.show()


# Lista de variables numéricas
numerical_vars = ['price', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365']

# Estadísticas descriptivas
print("Estadísticas Descriptivas:")
print(total_data[numerical_vars].describe())

# Histogramas y boxplots para cada variable
for var in numerical_vars:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histograma con KDE
    sns.histplot(total_data[var], kde=True, ax=axes[0])
    axes[0].set_title(f'Histograma de {var}')
    
    # Boxplot
    sns.boxplot(x=total_data[var], ax=axes[1])
    axes[1].set_title(f'Boxplot de {var}')
    
#-------------------------------------------------------------------------------------------------------------------------------
# Analisis de variables multivariantes.
# Al enfocarnos en la variable que mas interesa en casos asi, que es el precio, se pueden analizar sus relaciones 
# con otras variables numéricas y categóricas para identificar patrones o correlaciones que puedan ser útiles para el
#  modelado predictivo o para entender mejor los factores que influyen en el precio de los listados.

# 1. ANÁLISIS NUMÉRICO-NUMÉRICO: Correlaciones entre variables numéricas
print("="*60)
print("1. CORRELACIONES NUMÉRICAS")
print("="*60)

numerical_vars = ['price', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365', 'latitude', 'longitude']

# Matriz de correlación
correlation_matrix = total_data[numerical_vars].corr()
print("\nMatriz de Correlación:")
print(correlation_matrix)

# Visualizar heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
plt.title('Matriz de Correlación - Variables Numéricas')
plt.tight_layout()
plt.show()

# Correlaciones con 'price' (la variable de interés)
print("\nCorrelaciones con 'price':")
price_corr = correlation_matrix['price'].sort_values(ascending=False)
print(price_corr)


# 2. ANÁLISIS CATEGÓRICO-CATEGÓRICO: Relaciones entre variables categóricas
print("\n" + "="*60)
print("2. ANÁLISIS CATEGÓRICO-CATEGÓRICO")
print("="*60)

categorical_vars = ['neighbourhood_group', 'room_type']

# Tabla de contingencia
contingency_table = pd.crosstab(total_data['neighbourhood_group'], total_data['room_type'])
print("\nTabla de Contingencia (neighbourhood_group vs room_type):")
print(contingency_table)

# Visualizar con stacked bar plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Stacked bar plot (recuento)
contingency_table.plot(kind='bar', stacked=True, ax=axes[0])
axes[0].set_title('Tipos de Habitación por Grupo de Vecindario')
axes[0].set_xlabel('Grupo de Vecindario')
axes[0].set_ylabel('Cantidad')
axes[0].legend(title='Tipo de Habitación')
axes[0].tick_params(axis='x', rotation=45)

# Stacked bar plot (porcentaje)
contingency_pct = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
contingency_pct.plot(kind='bar', stacked=True, ax=axes[1])
axes[1].set_title('Tipos de Habitación por Grupo de Vecindario (%)')
axes[1].set_xlabel('Grupo de Vecindario')
axes[1].set_ylabel('Porcentaje (%)')
axes[1].legend(title='Tipo de Habitación')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Chi-square test
from scipy.stats import chi2_contingency
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"\nPrueba Chi-Square:")
print(f"Chi-square: {chi2:.4f}")
print(f"P-value: {p_value:.4e}")
print(f"Conclusión: {'Hay relación significativa' if p_value < 0.05 else 'No hay relación significativa'} entre las variables")


# 3. ANÁLISIS NUMÉRICO-CATEGÓRICO: Relaciones entre precio y variables categóricas
print("\n" + "="*60)
print("3. ANÁLISIS NUMÉRICO-CATEGÓRICO (PRECIO vs CATEGÓRICAS)")
print("="*60)

# Price by room_type
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Boxplot: Price by room_type
sns.boxplot(data=total_data, x='room_type', y='price', ax=axes[0,0])
axes[0,0].set_title('Precio por Tipo de Habitación')
axes[0,0].set_ylabel('Precio ($)')

# Violin plot: Price by neighbourhood_group
sns.violinplot(data=total_data, x='neighbourhood_group', y='price', ax=axes[0,1])
axes[0,1].set_title('Precio por Grupo de Vecindario')
axes[0,1].set_ylabel('Precio ($)')
axes[0,1].tick_params(axis='x', rotation=45)

# Barplot: Average price by room_type
avg_price_room = total_data.groupby('room_type')['price'].mean().sort_values(ascending=False)
sns.barplot(x=avg_price_room.index, y=avg_price_room.values, ax=axes[1,0])
axes[1,0].set_title('Precio Promedio por Tipo de Habitación')
axes[1,0].set_ylabel('Precio Promedio ($)')
axes[1,0].set_xlabel('Tipo de Habitación')

# Barplot: Average price by neighbourhood_group
avg_price_neighbourhood = total_data.groupby('neighbourhood_group')['price'].mean().sort_values(ascending=False)
sns.barplot(x=avg_price_neighbourhood.index, y=avg_price_neighbourhood.values, ax=axes[1,1])
axes[1,1].set_title('Precio Promedio por Grupo de Vecindario')
axes[1,1].set_ylabel('Precio Promedio ($)')
axes[1,1].set_xlabel('Grupo de Vecindario')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Estadísticas descriptivas de precio por categorías
print("\nPrecio por Tipo de Habitación:")
print(total_data.groupby('room_type')['price'].describe())

print("\nPrecio por Grupo de Vecindario:")
print(total_data.groupby('neighbourhood_group')['price'].describe())

# ANOVA: Verificar si hay diferencias significativas en precio entre grupos
from scipy.stats import f_oneway

price_by_room = [group['price'].values for name, group in total_data.groupby('room_type')]
f_stat, p_value = f_oneway(*price_by_room)
print(f"\nANOVA Test (Price by Room Type):")
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4e}")
print(f"Conclusión: {'Hay diferencias significativas' if p_value < 0.05 else 'No hay diferencias significativas'} en precio entre tipos de habitación")

price_by_neighbourhood = [group['price'].values for name, group in total_data.groupby('neighbourhood_group')]
f_stat, p_value = f_oneway(*price_by_neighbourhood)
print(f"\nANOVA Test (Price by Neighbourhood Group):")
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4e}")
print(f"Conclusión: {'Hay diferencias significativas' if p_value < 0.05 else 'No hay diferencias significativas'} en precio entre grupos de vecindarios")


#----------------------------------------------------------------------------------------------------------------------


#Ingenieria de caracteristicas
total_data.describe()

fig, axes = plt.subplots(3, 3, figsize = (15, 15))

sns.boxplot(ax = axes[0, 0], data = total_data, y = "neighbourhood_group")
sns.boxplot(ax = axes[0, 1], data = total_data, y = "price")
sns.boxplot(ax = axes[0, 2], data = total_data, y = "minimum_nights")
sns.boxplot(ax = axes[1, 0], data = total_data, y = "number_of_reviews")
sns.boxplot(ax = axes[1, 1], data = total_data, y = "calculated_host_listings_count")
sns.boxplot(ax = axes[1, 2], data = total_data, y = "availability_365")
sns.boxplot(ax = axes[2, 0], data = total_data, y = "room_type")

plt.tight_layout()

plt.show()


# TRATAMIENTO DE OUTLIERS CON IQR
print("="*70)
print("DETECTAR Y ELIMINAR OUTLIERS CON IQR (INTERQUARTILE RANGE)")
print("="*70)

# Variables numéricas a limpiar
numerical_vars_to_clean = ['price', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365']

# Almacenar información de outliers
outliers_info = {}
total_data_clean = total_data.copy()

for var in numerical_vars_to_clean:
    print(f"\n{'='*70}")
    print(f"Variable: {var.upper()}")
    print(f"{'='*70}")
    
    # Estadísticas
    stats = total_data[var].describe()
    print(f"\nEstadísticas descriptivas:")
    print(stats)
    
    # IQR
    Q1 = stats["25%"]
    Q3 = stats["75%"]
    iqr = Q3 - Q1
    
    # Límites
    upper_limit = Q3 + 1.5 * iqr
    lower_limit = Q1 - 1.5 * iqr
    
    print(f"\nIQR Calculation:")
    print(f"Q1 (25%): {Q1}")
    print(f"Q3 (75%): {Q3}")
    print(f"IQR (Q3 - Q1): {iqr}")
    print(f"Upper limit: {upper_limit}")
    print(f"Lower limit: {lower_limit}")
    
    # Contar outliers
    outliers_upper = (total_data[var] > upper_limit).sum()
    outliers_lower = (total_data[var] < lower_limit).sum()
    total_outliers = outliers_upper + outliers_lower
    
    print(f"\nOutliers detected:")
    print(f"Above upper limit ({upper_limit}): {outliers_upper}")
    print(f"Below lower limit ({lower_limit}): {outliers_lower}")
    print(f"Total outliers: {total_outliers}")
    
    # Valores especiales (0, 1)
    count_0 = (total_data[var] == 0).sum()
    count_1 = (total_data[var] == 1).sum()
    print(f"\nSpecial values:")
    print(f"Count of 0: {count_0}")
    print(f"Count of 1: {count_1}")
    
    # Almacenar info
    outliers_info[var] = {
        'upper_limit': upper_limit,
        'lower_limit': lower_limit,
        'outliers': total_outliers,
        'iqr': iqr
    }
    
    # Eliminar outliers
    total_data_clean = total_data_clean[
        (total_data_clean[var] <= upper_limit) & 
        (total_data_clean[var] >= lower_limit)
    ]

print(f"\n{'='*70}")
print("SUMMARY OF OUTLIER REMOVAL")
print(f"{'='*70}")
print(f"Original rows: {len(total_data)}")
print(f"Rows after removing outliers: {len(total_data_clean)}")
print(f"Rows removed: {len(total_data) - len(total_data_clean)}")
print(f"Percentage removed: {((len(total_data) - len(total_data_clean)) / len(total_data) * 100):.2f}%")

print(f"\nOutliers summary by variable:")
for var, info in outliers_info.items():
    print(f"  {var}: {info['outliers']} outliers found")

print(f"\nDataset after cleaning:")
print(f"Shape: {total_data_clean.shape}")
print(f"Columns: {total_data_clean.columns.tolist()}")

# NORMALIZACIÓN DE CARACTERÍSTICAS CON MinMaxScaler
print("="*70)
print("NORMALIZACIÓN DE VARIABLES NUMÉRICAS (MinMax Scaler)")
print("="*70)

from sklearn.preprocessing import MinMaxScaler

# Variables numéricas a normalizar
numerical_variables = ['number_of_reviews', 'minimum_nights', 'calculated_host_listings_count', 
                       'availability_365', 'latitude', 'longitude']

# Variables categóricas (se mantienen sin cambios)
categorical_variables = ['neighbourhood_group', 'room_type']

# Crear el scaler
scaler = MinMaxScaler()

# Aplicar normalización solo a variables numéricas
scaled_features = scaler.fit_transform(total_data_clean[numerical_variables])

# Crear dataframe con variables normalizadas
df_scaled = pd.DataFrame(scaled_features, 
                        index=total_data_clean.index, 
                        columns=numerical_variables)

# Agregar variables categóricas sin normalizar
for cat_var in categorical_variables:
    df_scaled[cat_var] = total_data_clean[cat_var].values

# Agregar la variable target (price)
df_scaled['price'] = total_data_clean['price'].values

print("\nVariables numéricas normalizadas:")
for var in numerical_variables:
    print(f"  {var}")

print(f"\nVariables categóricas (sin cambios):")
for var in categorical_variables:
    print(f"  {var}")

print(f"\nVariable target:")
print(f"  price")

print(f"\n{'='*70}")
print("DATASET NORMALIZADO - RESUMEN")
print(f"{'='*70}")
print(f"Forma del dataset: {df_scaled.shape}")
print(f"Columnas: {df_scaled.columns.tolist()}")
print(f"\nPrimeras 5 filas:")
print(df_scaled.head())

print(f"\nEstadísticas de variables normalizadas (rango 0-1):")
print(df_scaled[numerical_variables].describe())

# SELECCIÓN DE CARACTERÍSTICAS CON SelectKBest Y CHI2
print("="*70)
print("SELECCIÓN DE CARACTERÍSTICAS CON CHI2 Y SelectKBest")
print("="*70)

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Preparar X y y
X = df_scaled.drop("price", axis=1).copy()
y = df_scaled["price"].copy()

# Codificar variables categóricas (chi2 requiere valores numéricos)
for cat_var in ['neighbourhood_group', 'room_type']:
    le = LabelEncoder()
    X[cat_var] = le.fit_transform(X[cat_var])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SelectKBest con chi2
selection_model = SelectKBest(chi2, k=4)
selection_model.fit(X_train, y_train)

ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns=X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns=X_test.columns.values[ix])

print(f"\nCaracterísticas seleccionadas: {X_train_sel.columns.tolist()}")
print(f"X_train_sel shape: {X_train_sel.shape}")
print(f"X_test_sel shape: {X_test_sel.shape}")
print(f"\nX_train_sel (primeras 5 filas):")
print(X_train_sel.head())

#---------------------------------------------------------------------------------------------

# GUARDAR DATASETS PROCESADOS
print("="*70)
print("GUARDAR DATASETS PROCESADOS EN CSV")
print("="*70)

# Agregar variable target (price) a los datasets
X_train_sel["price"] = list(y_train)
X_test_sel["price"] = list(y_test)

print(f"\nDatasets con variable target agregada:")
print(f"X_train_sel shape: {X_train_sel.shape}")
print(f"X_test_sel shape: {X_test_sel.shape}")

# Guardar en CSV
X_train_sel.to_csv("../data/processed/clean_train.csv", index=False)
X_test_sel.to_csv("../data/processed/clean_test.csv", index=False)

print(f"\nArchivos guardados exitosamente:")
print(f"  ✓ ../data/processed/clean_train.csv")
print(f"  ✓ ../data/processed/clean_test.csv")

print(f"\nÚltimas filas de clean_train.csv:")
print(X_train_sel.tail())