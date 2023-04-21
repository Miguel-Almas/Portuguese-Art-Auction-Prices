import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
from datetime import datetime, timedelta
import sys
import os
import xgboost
import joblib
from wordcloud import WordCloud
from PIL import Image
SEED = 101

def get_predictions(df,dim_1,dim_2,birth_date,death_date,artist,technique):
  dim_1 = dim_1
  dim_2 = dim_2
  dim_3 = 1

  if birth_date is not None:
    birth_date = birth_date
  else:
    birth_date = 'no information'
  if death_date is not None:
    death_date = death_date
  else:
    death_date = 'no information'
  if artist is not None and len(artist) != 0:
    artist = artist
  else:
    artist = 'no information'
  if technique is not None and len(technique) != 0:
    technique = technique
  else:
    technique = []

  #Make dictionary from values
  dic_row = {'Number of Authors':1,'Number of Artworks':1,'1 Author':str.lower(artist),'1 Author Birth':str(birth_date),
           '1 Author Death':str(death_date),'Dim 1':dim_1,'Dim 2':dim_2,'Dim 3':dim_3,'Year':datetime.today().year,
           'Technique':'/'.join(technique)}

    #Create DataFrame
  df_inserted = pd.DataFrame(dic_row,index=[0])

  #Deal with dimensions
  df_inserted['Dim 1'] = df_inserted['Dim 1'].fillna(1)
  df_inserted['Dim 2'] = df_inserted['Dim 2'].fillna(1)
  df_inserted['Dim 3'] = df_inserted['Dim 3'].fillna(1)

  df_inserted['Area'] = df_inserted['Dim 1'] * df_inserted['Dim 2']# * df_inserted['Dim 3']

  if df_inserted['Area'].iloc[0] == 1:
      df_inserted['Area'] = df['Area'].mean()
      
  df_inserted['Shape'] = np.nan
  df_inserted.loc[(df_inserted['Dim 1'].astype('float64') > df_inserted['Dim 2'].astype('float64')),'Shape'] = 'Horizontal'
  df_inserted.loc[(df_inserted['Dim 1'].astype('float64') < df_inserted['Dim 2'].astype('float64')),'Shape'] = 'Vertical'
  df_inserted.loc[(df_inserted['Dim 1'].astype('float64') == df_inserted['Dim 2'].astype('float64')),'Shape'] = 'Square'

  #Deal with Year of Birth and Death
  year_decade_map = {}
  first_year=1800
  first_decade = 1800
  for i in range(23):
      for j in range(10):
          year_decade_map[str(first_year)] = str(first_decade)+'-'+str(first_decade+10)
          first_year +=1
      first_decade +=10
    
  df_inserted['1 Author Birth Decade'] = df_inserted['1 Author Birth'].map(year_decade_map)
  df_inserted['1 Author Death Decade'] = df_inserted['1 Author Death'].map(year_decade_map)

  df_inserted['1 Author Birth Decade'] = df_inserted['1 Author Birth Decade'].fillna('no information')
  df_inserted['1 Author Death Decade'] = df_inserted['1 Author Death Decade'].fillna('no information')
    
  path = './assets/models/'
  with open(path+'author_birth_decade.npy', 'rb') as f:
      list_decades_birth = list(np.load(f,allow_pickle=True))
    
  with open(path+'author_death_decade.npy', 'rb') as f:
      list_decades_death = list(np.load(f,allow_pickle=True))

  for i in list_decades_birth:
      if df_inserted['1 Author Birth Decade'].iloc[0] in i:
          df_inserted['1 Author Birth Decade_'+i] = 1
      else:
          df_inserted['1 Author Birth Decade_'+i] = 0
      
  for i in list_decades_death:
      if df_inserted['1 Author Death Decade'].iloc[0] in i:
          df_inserted['1 Author Death Decade_'+i] = 1
      else:
          df_inserted['1 Author Death Decade_'+i] = 0
          
  #Year
  list_years = ['Year_2020.0','Year_2019.0','Year_2018.0','Year_2017.0']
  for i in list_years:
      if str(df_inserted['Year'].iloc[0]) in i:
          df_inserted[i] = 1
      else:
          df_inserted[i] = 0
          
  #Technique   
  with open(path+'techniques.npy', 'rb') as f:
      list_techniques = list(np.load(f,allow_pickle=True))  

  for i in list_techniques:
      if i in df_inserted['Technique'].iloc[0]:
          df_inserted[i] = 1
      else:
          df_inserted[i] = 0
          
  #Shape
  list_shape = ['Horizontal','Vertical','Square']
  for i in list_shape:
      if str(df_inserted['Shape'].iloc[0]) in i:
          df_inserted['Shape_'+i] = 1
      else:
          df_inserted['Shape_'+i] = 0        

  #Mean Encodings
  df_map_author = pd.read_csv('./assets/models/author_mapping_mean_encodings.csv')
  df_map_colours = pd.read_csv('./assets/models/colours_mapping_mean_encodings.csv')

  df_inserted['1 Author_mean_encoded'] = df_inserted['1 Author'].map(df_map_author.set_index('1 Author').to_dict()['median'])
  df_inserted['Dominant Colour Name_mean_encoded'] = df['Final Price'].mean()
  #df_inserted['Dominant Colour Name_mean_encoded'] = df_inserted['Dominant Colour Name'].map(df_map_colours.set_index('Dominant Colour Name').to_dict()['median'])

  #Features to drop from feature importance
  with open(path+'feats_drop.npy', 'rb') as f:
      list_feats_drop = list(np.load(f,allow_pickle=True))  
  
  #Scaling 
  col_drop = ['1 Author','1 Author Birth','1 Author Death','Dim 1','Dim 2','Dim 3','Year','Technique','1 Author Birth Decade','1 Author Death Decade','Shape']
  scaler = joblib.load('./assets/models/scaler.bin')

  #Ordered columns
  with open(path+'features_ordered.npy', 'rb') as f:
      list_reorder = list(np.load(f,allow_pickle=True))  

  df_final = df_inserted.drop(col_drop+list_feats_drop,axis=1)[list_reorder] #scaler.transform(df_inserted.drop(col_drop+list_feats_drop,axis=1)[list_reorder])
  print(df_final.columns)
  #Make prediction
  #prediction = np.exp(model_xgb.predict(df_final))[0]
  prediction = np.exp(joblib_model.predict(df_final.to_numpy()))[0]

  return 'Predicted auction price:\n{:.2f} €'.format(prediction)

#to get the current working directory
directory = os.getcwd()
sys.path.append(directory+'/assets')

#Import the DataFrame
df_art = pd.read_csv('./assets/data/data_cml_processed.csv')
#Import model
model_xgb = xgboost.XGBRegressor()
model_xgb.load_model('./assets/models/model.bin')
joblib_model = joblib.load('./assets/models/ridge_model.pkl')

#Get Series with Artist Name Frequency
di = df_art[df_art['1 Author'] != 'no information']['1 Author'].value_counts()
#Create mask with gavel shape for wordcloud
gavel_mask = np.array(Image.open('./assets/images/auction-hammer-symbol.jpg'))
#Create wordcloud
wc = WordCloud(background_color='white',colormap='Dark2',mask=gavel_mask).generate_from_frequencies(frequencies=di)
wc_img = wc.to_image()
wc_img.save("./assets/images/hammer.png","PNG")

#Lists for dropdown
list_decades = []
list_authors = ['No information'] + list(df_art[df_art['1 Author'] != 'no information']['1 Author'].str.title().value_counts().index)
list_techniques = ['assinar', 'papel', 'datar', 'numerada', 'serigrafia', 'tecnica',
         'misturar', 'tela','marcar', 'defeito', 'oleo', 'pequeno', 'sinal',
         'uso','europeu', 'decoracao','metal', 'vidro', 'verso', 'acrilico',
         'madeira', 'escultura', 'gravura', 'identificar', 'português', 
         'tinta', 'policromada', 'china', 'fabricar', 'aguarela', 'assinatura',    
         'azulejo', 'falta', 'mancha', 'portugue', '"', 'cromado', 'italiano',  'material', 
         'platex',  'dourar','cristal',  'pintado', 'autor', 
         'ceramica','humidade','colagem','prova','prateado', 'hc']

dic_techniques = { 
        'assinada':'assinar', 'papel':'papel', 'datada':'datar', 'numerada':'numerada', 'serigrafia':'serigrafia', 'tecnica':'tecnica',
        'com marcas':'marcar', 'tela':'tela', 'misturar':'misturar', 'defeito':'defeito', 'pequeno':'pequeno', 'oleo':'oleo', 'sinais':'sinal',
        'europeu':'europeu', 'uso':'uso', 'metal':'metal', 'decorada':'decoracao', 'vidro':'vidro', 'verso':'verso', 'acrilico':'acrilico',
        'madeira':'madeira', 'escultura':'escultura', 'identificada':'identificar', 'português':'português', 'gravura':'gravura',
        'policromada':'policromada', 'assinatura':'assinatura', 'aguarela':'aguarela', 'tinta':'tinta', 'china':'china', 'fabricar':'fabricar',
        'falta':'falta', 'cromada':'cromado', 'italiano':'italiano', 'portugue':'portugue', 'material':'material', 'azulejo':'azulejo',
        'platex':'platex', 'mancha':'mancha', 'cristal':'cristal', 'dourada':'dourar', '"':'"','autor':'autor', 'pintado':'pintado',
        'prateado':'prateado', 'plastico':'plastico', 'prova':'prova', 'hc':'hc', 'relevo':'relevado', 'base':'base'
        }

st.image(wc_img,width = 480)
st.header('Portuguese Art Auction Forecaster')

#Dropdown for the model explorer
col1, col2 = st.columns(2)

with col1:
    ##Artist
    chosen_artist = st.selectbox(
        'Select one artist',
        list_authors)
    ##Artwork Width
    chosen_art_width = st.slider('''Input the Artwork's Width (cm)''',min_value=1, max_value=500, value=100)
    ##Artist Birth Date
    chosen_artist_birth_date = st.number_input('''Insert the Artist's Year of Birth''',min_value=1870,max_value=2023,value=1950,step=1,format='%i')
with col2:
    ##Techniques
    chosen_techniques = st.multiselect(
        'Select Techniques',
        list_techniques)
    ##Artwork Height
    chosen_art_height = st.slider('''Input the Artwork's Height (cm)''',min_value=1, max_value=500, value=100)
    ##Artist Death Date
    col2_1, col2_2 = st.columns(2)
    with col2_1:
        #artist_has_passed_away = st.checkbox('Artist has Passed Away')
        artist_has_passed_away = st.selectbox(
            'Has the Artist Passed Away?',
            ('Yes', 'No'))
    if artist_has_passed_away == 'Yes':
        with col2_2:
            chosen_artist_death_date = st.number_input('''Insert the Artist's Year of Death''',min_value=1870,max_value=2023,value=1950,step=1,format='%i')
    else:
        chosen_artist_death_date = np.nan

#Get predictions
pred = get_predictions(df_art,
                       (chosen_art_width),
                       (chosen_art_height),
                       (chosen_artist_birth_date),
                       (chosen_artist_death_date),
                       (chosen_artist),
                       (chosen_techniques)
)

st.header('Predicted Price:', anchor=None)
st.text(pred)

