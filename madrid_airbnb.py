#-----librerías----------------------------------------------------------------------------------------

### Importamos las librerias para trabajar
import os
import urllib.request
import shutil
import gzip
import pandas as pd
import numpy as np
import json
import io
import warnings
from unicodedata import name
warnings.simplefilter(action='ignore', category=FutureWarning)

### Mapas interactivos
# import folium
# from folium.plugins import FastMarkerCluster
# import geopandas as gpd
# from branca.colormap import LinearColormap

# Gráficos e imágenes
# import IPython.display as display
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.dates as mdates
from matplotlib import ticker
import plotly.express as px
import plotly.graph_objs as go
import plotly.graph_objects as go
from plotly.offline import iplot, init_notebook_mode
# import cufflinks
# cufflinks.go_offline(connected=True)
# init_notebook_mode(connected=True)

# Streamlit
import streamlit as st
import streamlit.components.v1 as components 


#-----lectura del dataset--------------------------------------------------------------------------
df = pd.read_csv('airbnb_anuncios.csv')

#-----configuracion de página--------------------------------------------------------------------------

st.set_page_config(page_title='AirBnB Madrid', layout='centered', page_icon='🌇')

#-----empieza la app-----------------------------------------------------------------------------------
st.markdown("<h1 style='text-align: center; '>Proyecto Módulo 2</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; '>Inside Airbnb: Madrid</h1>", unsafe_allow_html=True)
st.image('https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/madrid-03-1537270899.jpg?crop=0.890625xw:1xh;center,top&resize=1200:*')
st.text("Fuente imagen: elle.com")

st.markdown("")
st.markdown("")

#-----Configuración de bloques---------------------------------------------------------------------------
bloques = st.tabs(["Limpieza del Dataset", "Análisis Exploratorio", "Modelo Predictivo"])

#-----Dataset-----------
with bloques[0]:
    st.image('https://www.parsehub.com/blog/content/images/2019/08/scrape-airbnb-data.jpg')
    st.text("Fuente imagen: parsehub.com")
    st.markdown("<h2>Librerías utilizadas</h2>", unsafe_allow_html=True)
    st.code('''# Importamos las librerias para trabajar
import wget
import os
import urllib.request
import shutil
import gzip
import pandas as pd
import numpy as np
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Mapas interactivos
import folium
from folium.plugins import FastMarkerCluster
import geopandas as gpd
from branca.colormap import LinearColormap

# Gráficos e imágenes
import IPython.display as display
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import plotly.graph_objs as go
import chart_studio.plotly as py
from plotly.offline import iplot, init_notebook_mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

# Streamlit
import streamlit as st
import streamlit.components.v1 as components 

# ML models
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV, LassoLarsCV
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler''')

    st.markdown("")
    st.markdown("<h2>Tratamiento de los datos</h2>", unsafe_allow_html=True)

    # Visualizamos nulos con sus respectivos porcentajes
    missing_values = df.isnull().sum()
    missing_values = missing_values.to_frame().rename(columns = {0:'nulos'})
    missing_values['porcentaje'] = df.isnull().sum()*100/df.shape[0]
    missing_values = missing_values.sort_values(by = ['nulos'], ascending = False)
    missing_values = missing_values[missing_values['porcentaje'] > 0]
    
    st.code('''# Visualizamos nulos con sus respectivos porcentajes
missing_values = df.isnull().sum()
missing_values = missing_values.to_frame().rename(columns = {0:'nulos'})
missing_values['porcentaje'] = df.isnull().sum()*100/df.shape[0]
missing_values = missing_values.sort_values(by = ['nulos'], ascending = False)
missing_values = missing_values[missing_values['porcentaje'] > 0]''')

    st.dataframe(missing_values)

    st.code('''# Procedemos a eliminamos las columnas deseadas
df = df.drop(['name', 'host_name', 'last_review'], axis=1)''')

    # Reemplazamos los valores nulos de 'reviews_per_month' por zero
    df = df.fillna(value={'reviews_per_month': 0})
    st.code('''# Ponemos a zero los valores nulos de 'reviews_per_month'
df = df.fillna(value={'reviews_per_month': 0})''')


#-----Información-----------
with bloques[1]:       
    st.image('https://blog.datawrapper.de/wp-content/uploads/2022/08/colorblind-f2-copy-1024x512.png')
    st.text("Fuente imagen: datawrapper.de")
    st.markdown("<h2>Análisis exploratorio</h2>", unsafe_allow_html=True)
    
#---grafico 1-----------------------
    st.markdown("<h3>Distrito</h3>", unsafe_allow_html=True)
    st.text("")

    st.markdown("**Mostramos un gráfico de la cantidad de airbnbs según los distritos de Madrid**")
    i1 = Image.open('i1.png')
    st.image(i1)
    st.markdown('Vemos que la gran mayoria de airbnbs estan en el **centro**.')

    map = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='neighbourhood_group',
                        title='Mostramos los datos anteriores con un gráfico de dispersión en un mapa de Madrid',
                        size_max=15, zoom=10, height=600, color_continuous_scale='plasma')
    map.update_layout(mapbox_style='open-street-map')
    st.plotly_chart(map)

#---grafico 2-----------------------
    st.markdown("<h3>Vecindario</h3>", unsafe_allow_html=True)
    st.text("")

    st.markdown("**Ahora mostramos un gráfico de la cantidad de airbnbs según los vecindarios de Madrid**")
    i2 = Image.open('i2.png')
    st.image(i2)
    st.markdown('Visto también el gráfico anterior, los vecindarios de **Embajadores, Universidad, Palacio, Sol, Justicia y Cortes** seguramente pertenecen al **centro**.')

    map = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='neighbourhood',
                        title='Mostramos también los datos anteriores con un gráfico de dispersión en un mapa de Madrid',
                        size_max=15, zoom=10, height=600, color_continuous_scale='plasma')
    map.update_layout(mapbox_style='open-street-map')
    st.plotly_chart(map)

#---grafico 3-----------------------
    st.markdown("<h3>Tipo de alojamiento</h3>", unsafe_allow_html=True)
    st.text("")

    st.markdown("**Ahora mostramos un gráfico de la cantidad de airbnbs según el tipo de alojamiento**")
    i3 = Image.open('i3.png')
    st.image(i3)
    st.markdown('Vemos que la mayoria de airbnbs de Madrid son **apartamentos o casas enteras** seguidos de **habitaciones privadas**.')

    map = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='room_type',
                        title='Mostramos también los datos anteriores con un gráfico de dispersión en un mapa de Madrid',
                        size_max=15, zoom=10, height=600, range_color='blues')
    map.update_layout(mapbox_style='open-street-map')
    st.plotly_chart(map)

#---grafico 4-----------------------
    st.markdown("<h3>Disponibilidad</h3>", unsafe_allow_html=True)
    st.text("")

    st.markdown("**Ahora mostramos un gráfico boxplot de la disponibilidad anual de los airbnb de la ciudad**")
    i4 = Image.open('i4.png')
    st.image(i4)
    st.markdown('''**El gráfico boxplot muestra la disponibilidad o cantidad de días en los que se puede alquilar un airbnb en los distintos vecindarios.**
    
**Vemos que el barrio más solicitado o el que tiene menos disponibilidad es el barrio de `Arganzuela`.**
**Y el barrio menos solicitado o que tiene más disponibilidad es `Vicálvaro`.**''')

    map = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='availability_365',
                        title='Mostramos también los datos anteriores con un gráfico de dispersión en un mapa de Madrid',
                        size_max=15, zoom=10, height=600)
    map.update_layout(mapbox_style='open-street-map')
    st.plotly_chart(map)


#-----Modelos-----------
with bloques[2]:
    st.image('https://tarikatechnologies.com/storage/2020/07/Header-4.jpg')
    st.text("Fuente imagen: tarikatechnologies.com")
    st.markdown("<h2>Modelo predictivo</h2>", unsafe_allow_html=True)

    st.markdown("<h3>Preparación del modelo</h3>", unsafe_allow_html=True)

    # Borramos las columnas que creemos que no son necesarias para nuestro modelo
    st.code('''# Creamos una copia de nuestro dataframe
df2 = df''')
    
    # Creamos una nueva variable 'price_range' para facilitar la predicción del modelo
    st.code('''# Creamos una nueva variable 'price_range' para facilitar la predicción del modelo
 
# Creamos una lista de condiciones
condiciones = [
    (df2['price'] < 50),
    (df2['price'] >= 50) & (df2['price'] < 75),
    (df2['price'] >= 75) & (df2['price'] < 100),
    (df2['price'] >= 100) & (df2['price'] < 125),
    (df2['price'] >= 125) & (df2['price'] < 150),
    (df2['price'] >= 150) & (df2['price'] < 175),
    (df2['price'] >= 175) & (df2['price'] < 200),
    (df2['price'] >= 200) & (df2['price'] < 225),
    (df2['price'] >= 225) & (df2['price'] < 250),
    (df2['price'] >= 250)]

# Creamos una lista de valores en funcion de las condiciones anteriores
valores = ['0-50', '50-75', '75-100', '100-125', '125-150', '150-175', '175-200', '200-225', '225-250', '>250']

# Creamos una nueva columna y le asignamos las condiciones y los valores anteriores con np.select
df2['price_range'] = np.select(condiciones, valores)''')

    # Borramos las columnas que creemos que no son necesarias para nuestro modelo
    st.code('''# Borramos las columnas que creemos que no son necesarias para nuestro modelo
df2 = df2.drop(['host_id', 'reviews_per_month','latitude', 'longitude', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365'], axis=1)''')

    # Transformamos algunas columnas con el LabelEncoder
    st.code('''# Transformamos algunas columnas con el LabelEncoder
labelencoder = LabelEncoder()
df2['neighbourhood_group'] = labelencoder.fit_transform(df2['neighbourhood_group'])
df2['neighbourhood'] = labelencoder.fit_transform(df2['neighbourhood'])
df2['room_type'] = labelencoder.fit_transform(df2['room_type'])
df2['price_range'] = labelencoder.fit_transform(df2['price_range'])''')

    # Creamos una matriz de correlación doble (Pearson y Spearman)
    st.code('''# Creamos una matriz de correlación doble (Pearson y Spearman)
Pearson_matrix = df2.loc[:, df2.columns != 'id'].corr(method = 'pearson')
Spearman_matrix = df2.loc[:, df2.columns != 'id'].corr(method = 'spearman')

fig, ax =plt.subplots(1, 2, figsize=(18,6))
sns.heatmap(Pearson_matrix, annot=True, annot_kws={"size": 10, 'rotation': 30}, vmin=-1.0, vmax=1.0, center=0, square=True, cmap="magma", linewidths=0.1, fmt='.2f', ax=ax[0])
sns.heatmap(Spearman_matrix, annot=True, annot_kws={"size": 10, 'rotation': 30}, vmin=-1.0, vmax=1.0, center=0, square=True, cmap="magma", linewidths=0.1, fmt='.2f', ax=ax[1])
ax[0].set_title('Pearson')
ax[1].set_title('Spearman')
fig.show();''')

    i5 = Image.open('i5.png')
    st.image(i5)

    st.markdown('Vemos que tanto la matriz de correlación de Pearson como de Spearman muestran unas dependencias entre variables muy pobres excepto entre `Price` y `Room_type`.')
    st.markdown("")

    # Seleccionamos la variable predictora X y la variable de respuesta Y
    st.code('''# Seleccionamos la variable predictora X y la variable de respuesta Y 
X = df2.loc[:, ~df2.columns.isin(['price', 'id'])].values
Y = df2['price'].values''')

    #st.markdown("")
    #st.markdown("")

    #st.markdown("<h3>Regresión lineal múltiple</h3>", unsafe_allow_html=True)
    #i8 = Image.open('i8.png')
    #st.image(i8)

    st.markdown("")
    st.markdown("")

    st.markdown("<h3>Vecinos cercanos KNN</h3>", unsafe_allow_html=True)
    i7 = Image.open('i7.png')
    st.image(i7)

    # Calculamos el valor de K (distancia vecinos) que mejor "accuracy" tiene
    st.code('''# Calculamos el valor de K (distancia vecinos) que mejor precisión tiene
score_list = []
k_list = []

# Vamos a probar con K del 1 al 20 de dos en dos
for k in list(range(1, 20, 2)):
  knn_model = KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="minkowski")
  score = cross_val_score(knn_model, X, Y, cv=3, scoring="accuracy")
  k_list.append(k)
  score_list.append(score.mean())

df_scores = pd.DataFrame({"K":k_list, 
                   "score":score_list
                   })

df_scores.set_index("K")["score"].plot(label="KNN Score")
plt.legend()
plt.show();''')

    i6 = Image.open('i6.png')
    st.image(i6)

    st.markdown("<h3>Conclusiones del modelo</h3>", unsafe_allow_html=True)
    st.markdown('''Como conclusiones vemos que:
* Nuestro modelo tiene la mejor precisión con una k=20 pero no es tanta la diferencia respecto k=7
* Solo obtenemos un 44% de precisión en el mejor de los casos.
* Podriamos probar distintos modelos predictivos de clasificación
* Llevar a cabo mejores técnicas de entrenamiento como la validación cruzada.
    ''')

    # st.markdown("<h3>Árbol de decisión</h3>", unsafe_allow_html=True)
