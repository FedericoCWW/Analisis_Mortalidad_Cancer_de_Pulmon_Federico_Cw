#!/usr/bin/env python
# coding: utf-8

# In[40]:


#Librerias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[2]:


#https://www.kaggle.com/datasets/masterdatasan/lung-cancer-mortality-datasets-v2


# In[3]:


df = pd.read_csv("lung_cancer_mortality_data_large_v2.csv")


# In[4]:


df.head()


# In[5]:


# Mostrar solo tratamiento de Cancer Estado 1 que sobrevivio
df_filt1 = df[(df["cancer_stage"]== "Stage I") & (df["survived"] == 1)]
#df_filt1 = df_filt1[df_filt1["survived"] == 1]


# In[6]:


#Aqui se imprime el df filtrado
df_filt1


# In[7]:


df_agrupado_tratamiento = df.groupby('treatment_type').size().reset_index(name='count')


# In[8]:


df_agrupado_tratamiento


# In[9]:


#convertir fechas a date time
df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'])
df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'])


# In[10]:


#Tambien las edades a Int por que medio al dope que sean floats
df['age'] = df['age'].astype("int")


# In[11]:


df['cholesterol_bmi_ratio'] = df['cholesterol_level'] / df['bmi']


# In[12]:


def categorizar_bmi(bmi):
    if bmi < 18.5:
        return("Underweight")
    elif bmi < 24.9:
        return("Normal Weight")
    elif bmi < 29.9:
        return("Overweight")
    elif bmi >= 30:
        return("Obese")


# In[13]:


df["bmi_categorized"] = df["bmi"].apply(categorizar_bmi)


# In[14]:


# Creo que el tema de la limpieza ya estaria


# In[15]:


sns.histplot(df.head(100), x="gender", hue="cancer_stage")


# In[16]:


#Barplot con casos de cancer por pais

n = 70000

fig, ax = plt.subplots(figsize=(12, 8))
sns.countplot(data=df.head(n), 
              y='country',
              ax=ax,
              order=df.head(n)['country'].value_counts().index,
              palette='cubehelix',
              edgecolor='black',
              linewidth=0.5,
              legend=False,
              hue='country',
              )

ax.set_ylabel("Pais")
ax.set_xlabel("Casos por pais")
ax.set_title('Casos de cancer por pais', 
             fontsize=16, 
             fontweight='bold',
             pad=20)
plt.tight_layout()
plt.savefig("casos_de_cancer_por_pais.png")
plt.show()




# In[17]:


# Lmplot plot de...
g = sns.lmplot(data=df.head(20000), 
               x='age', 
               y='cholesterol_level',
               height=8,  # altura en pulgadas
               aspect=1.5,
               hue="gender")  # ancho = height * aspect

g.set_xlabels("Edad")
g.set_ylabels("Nivel de colesterol")


# Exportar
g.savefig('colesterol_vs_edad.png') 


# In[18]:


# estimador condicional de Kernel por hypertension	asthma	cirrhosis


# In[19]:


fig, ax = plt.subplots(figsize=(10, 8))

df_plot = df.head(100)
df_plot['condition_count'] = df_plot[['hypertension', 'asthma', 'cirrhosis']].sum(axis=1)
df_plot['condition_group'] = pd.cut(df_plot['condition_count'], 
                                     bins=[-1, 0, 1, 3], 
                                     labels=['Ninguna', '1 condición', '2+ condiciones'])

sns.kdeplot(data=df_plot, 
            x='bmi', 
            y='cholesterol_level',
            hue='condition_group',
            palette='viridis',
            alpha=0.6,
            thresh=0.05,
            ax=ax,
            fill=True)

ax.set_title('BMI vs colesterol categorizado por cantidad de condiciones', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('BMI (indice de masa corporal)', fontsize=12)
ax.set_ylabel('Nivel de colesterol', fontsize=12)
plt.tight_layout()
plt.show()


# In[20]:


fig, ax = plt.subplots(figsize=(10, 8))

# Crear una columna categórica combinada

sns.kdeplot(data=df.head(100), 
            x='age', 
            y='cholesterol_bmi_ratio',
            hue='smoking_status',
            palette='viridis',
            alpha=0.6,
            thresh=0.05,
            ax=ax,
            fill=True)

ax.set_title('BMI vs colesterol por condiciones medicas', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('BMI (indice de masa corporal)', fontsize=12)
ax.set_ylabel('Nivel de colesterol', fontsize=12)
plt.tight_layout()
plt.show()
plt.savefig("BMI_vs_colesterol_por_condiciones_medicas.png")


# In[21]:


df.head()


# In[30]:


features = ['age', 'bmi', 'smoking_status', 'gender', 'cancer_stage', 
            'hypertension', 'asthma', 'cirrhosis', 'survived']
target = 'cholesterol_level'
df_rl = pd.DataFrame.copy(df)


# In[31]:


X = df_rl[features]        #variable independendiente
y = df_rl[target]          #variable denpendiente


# In[32]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[34]:


# Identificar columnas numéricas y categóricas
numeric_features = ['age', 'bmi']
categorical_features = ['smoking_status', 'gender', 'cancer_stage']

# Crear preprocesadores
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)


# In[35]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# In[36]:


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])


# In[39]:


# entrenamiento
pipeline.fit(x_train, y_train)


# In[43]:


y_pred_train = pipeline.predict(x_train)
y_pred_test = pipeline.predict(x_test)

train_r2 = r2_score(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_mae = mean_absolute_error(y_train, y_pred_train)


# In[44]:


test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)


# In[49]:


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Valores reales vs predichos (prueba)
axes[0].scatter(y_test, y_pred_test, alpha=0.5, edgecolors='black', linewidth=0.5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Predicción perfecta')
axes[0].set_xlabel('Valores Reales', fontsize=12)
axes[0].set_ylabel('Valores Predichos', fontsize=12)
axes[0].set_title(f'Regresión Lineal: Nivel de colesterol\nR² = {test_r2:.3f}', 
                  fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Distribución de residuos
residuos = (y_test - y_pred_test)
axes[1].hist(residuos, bins=30, edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Residuos (Real - Predicho)', fontsize=12)
axes[1].set_ylabel('Frecuencia', fontsize=12)
axes[1].set_title('Distribución de Residuos', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.subplots_adjust(wspace=0.4)
plt.tight_layout(pad=3.0)
plt.savefig('regresion_lineal_resultados.png', dpi=300, bbox_inches='tight')
plt.show()



# In[ ]:




