import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import numpy as np
import pandas as pd
import xgboost
from sklearn.linear_model import Ridge
from category_encoders import LeaveOneOutEncoder
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import datetime
import joblib
from wordcloud import WordCloud
import base64
from PIL import Image
from io import BytesIO
import pathlib
from app import app
# import all pages in the app
from apps import auctions, artworks, home

SEED = 101
# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../assets").resolve()

#------------------------ Create a WordCount Image to display in Dash---------------------------------------------

df = pd.read_csv(".\\assets\\data_cml_processed.csv",index_col=0)
model_xgb = xgboost.XGBRegressor()
model_xgb.load_model('.\\assets\\model.bin')
joblib_model = joblib.load('.\\assets\\ridge_model.pkl')

#Get Series with Artist Name Frequency
di = df[df['1 Author'] != 'no information']['1 Author'].value_counts()
#Create mask with gavel shape for wordcloud
gavel_mask = np.array(Image.open('.\\assets\\auction-hammer-symbol.jpg'))
#Create wordcloud
wc = WordCloud(background_color='white',colormap='Dark2',mask=gavel_mask).generate_from_frequencies(frequencies=di)
wc_img = wc.to_image()
wc_img.save(".\\assets\\hammer.png","PNG")

with BytesIO() as buffer:
    wc_img.save(buffer, 'png')
    img_cloud = base64.b64encode(buffer.getvalue()).decode()


#Lists for dropdown
list_decades = []
list_authors = ['No information'] + list(df[df['1 Author'] != 'no information']['1 Author'].str.title().value_counts().index)
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

#Set the Layout
layout = html.Div([
    dbc.Container([
        dbc.Row([
          dbc.Col(
            html.Img(src = ".\\assets\\hammer.png",style={'width':'100%','height':'88%'}),style={'margin-top':'35px'} #style={'width':'500px','height':'320px'}
          ,width=6,xs=12,sm=12,md=6,lg=6,xl=6,),
          dbc.Col(
            dbc.Row([
              #html.H1("The Modern Portuguese Auctioned Art dashboard", className="text-center",style={
              #'textAlign': 'center','margin-left':'0px', 'margin-right':'0px'}),
              html.H2(children='Welcome to this dashboard!',style={
              'textAlign': 'left','margin-top':'0px','margin-left':'0px','margin-left':'20px', 'margin-right':'20px'}),
              html.H5(children='This app has two main focus: to allow price prediction based on user input features and exploration of artworks sold in'
                                     ' auctions, with filters by artist.',style={
              'textAlign': 'left','margin-top':'20px','margin-left':'0px','margin-left':'20px', 'margin-right':'20px'}),
              #dbc.Row(dbc.Col(
                dbc.Card(children=[html.H4(children='Start with the works of art...',className="text-center"),
                                      dcc.Link(dbc.Button('Explore artworks',color="primary",className="mt-3",style={'width':'100%'}), href='/artworks'),
                                       ],
                             body=True, color="dark", outline=True,style={'margin-top':'20px','margin-left':'20px', 'margin-right':'20px'}),#)),
            ]),
          className="mb-5 mt-5",width=6,xs=12,sm=12,md=6,lg=6,xl=6,),
        ]),
        dbc.Row([
          #html.H2(children='''
          #  Or
          #  ''',
          #  style={'textAlign': 'center','margin-top':'20px','margin-left':'0px', 'margin-right':'0px'}
          #),
          html.H4(children='''
            Or choose the parameters below to obtain a price predictions.
            ''',
            style={
              'textAlign': 'center','margin-top':'0px','margin-left':'0px', 'margin-right':'0px'
            }
          ),
          html.Div(children='''
            Please note that this capability is still in early stages and predictions are weak.
            ''',
            style={
              'textAlign': 'center',
            }
          ),
        ],style={'margin-top':'20px','margin-left':'0px', 'margin-right':'0px'}),
        dbc.Row([
          dbc.Col([
            html.Div(children='''
              Choose the artwork's artist. Use no information if unknown of artist not on list.
              ''',
              style={
                'textAlign': 'left','margin-top':'0px'
              }),
            dcc.Dropdown(id='artist-dropdown',
                              options=[{'label':i,'value':i} for i in list_authors],
                              placeholder="Select one artist",
                              value='no information',
                              multi=False,
                              style={"width": "100%", 'margin-left':'0px', 'margin-right':'0px'},
            )
          ]),
        ],style={'margin-top':'20px','margin-left':'0px', 'margin-right':'0px'}),
        dbc.Row([
          dbc.Col([
            html.Div(children='''
              Choose the artist's birth year.
              ''',
              style={
                'textAlign': 'left','margin-top':'0px'
              }),
            html.Div([
              dcc.Slider(
                id='slider-birth-date',
                min=1870,
                max=2020,
                step=1,
                #value=,
              ),
            html.Div(id='slider-output-birth-date'),
            ],style={'width': '100%','display': 'inline-block'}
            ),
          ],style={'margin-top':'0px','margin-left':'0px', 'margin-right':'0px'}),
          dbc.Col([
            html.Div(children='''
              Choose the artist's year of death.
              ''',
              style={
                'textAlign': 'left','margin-top':'0px'
              }),
            html.Div([
              dcc.Slider(
                id='slider-death-date',
                min=1870,
                max=2020,
                step=1,
                #value=,
              ),
            html.Div(id='slider-output-death-date'),
            ],style={'width': '100%','display': 'inline-block'}
            ),
          ],style={'padding':'00px'}),
        ],style={'margin-top':'20px','margin-left':'0px', 'margin-right':'0px'}),
        dbc.Row([
          dbc.Col([
            html.Div(children='''
              Dimension 1 - Width of artwork in cm. If unknown, leave as 1.
              ''',
              style={
                'textAlign': 'left',
              }),
            html.Div([
              dcc.Slider(
                id='slider-dim-1',
                min=1,
                max=200,
                step=1,
                value=1,
              ),
            html.Div(id='slider-output-dim-1'),
            ],style={'width': '100%','display': 'inline-block'}
            ),
          ]),
          dbc.Col([
            html.Div(children='''
              Dimension 2 - Height of artwork in cm. If unknown, leave as 1.
              ''',
              style={
                'textAlign': 'left',
              }),
            html.Div([
              dcc.Slider(
                id='slider-dim-2',
                min=1,
                max=200,
                step=1,
                value=1,
              ),
            html.Div(id='slider-output-dim-2'),
            ],style={'width': '100%','display': 'inline-block'}
            ),
          ]),
        ],style={'margin-top':'20px','margin-left':'0px', 'margin-right':'0px'}),
        dbc.Row([
          dbc.Col([
            html.Div(children='''
              Techniques - Select keyword characteristics. Note that words are lemmas.
              ''',
              style={
                'textAlign': 'left',
              }),
            dcc.Dropdown(id='technique-dropdown',
                              options=[{'label':i,'value':i} for i in list_techniques],
                              placeholder="Select one or more techniques",
                              multi=True,
                              style={"width": "100%", 'margin-left':'0px', 'margin-right':'0px'},
                  ),
          ]),
        ],style={'margin-top':'20px','margin-left':'0px', 'margin-right':'0px'}),
        dbc.Row([
          dbc.Col(
          html.H4(id='prediction',
              style={
                'textAlign': 'center',
              }),
          )],style={'margin-top':'40px','margin-bottom':'60px','margin-left':'0px', 'margin-right':'0px'}),
        dbc.Row([
            #dbc.Col(dbc.Card(children=[html.H3(children='Explore the artwork data collected.',
            #                                   className="text-center"),
            #                          dcc.Link(dbc.Button('Explore artworks',color="primary",className="mt-3",style={'width':'100%'}), href='/artworks'),
            #                           ],
            #                 body=True, color="dark", outline=True)
            #        , width=4, className="mb-4"),
            dbc.Col(dbc.Card(children=[html.H3(children='The Auction House website featured in this dashboard',
                                               className="text-center"),
                                        dbc.Button("Cabral Moncada Leilões", href="https://www.cml.pt/leiloes/online",color="primary",className="mt-3"),
                                       ],
                             body=True, color="dark", outline=True), width=6, className="mb-4"),
            dbc.Col(dbc.Card(children=[html.H3(children='Access the code used to support this dashboard',className="text-center"),
                                       dbc.Button("GitHub",
                                                  href="https://github.com/Miguel-Almas/Portuguese-Art-Auction-Prices",
                                                  color="primary",
                                                  className="mt-3"),
                                       ],
                             body=True, color="dark", outline=True), width=6, className="mb-4"),
        ], className="mb-5"),
    ])
])

@app.callback(
    dash.dependencies.Output('slider-output-dim-1', 'children'),
    [dash.dependencies.Input('slider-dim-1', 'value')])
def update_output(value):
    return 'You have selected "{}" cm'.format(value)

@app.callback(
    dash.dependencies.Output('slider-output-dim-2', 'children'),
    [dash.dependencies.Input('slider-dim-2', 'value')])
def update_output(value):
    return 'You have selected "{}" cm'.format(value)

@app.callback(
    dash.dependencies.Output('slider-output-birth-date', 'children'),
    [dash.dependencies.Input('slider-birth-date', 'value')])
def update_output(value):
    return 'You have selected the year "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('slider-output-death-date', 'children'),
    [dash.dependencies.Input('slider-death-date', 'value')])
def update_output(value):
    return 'You have selected the year "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('prediction', 'children'),
    [dash.dependencies.Input('slider-dim-1', 'value'),
    dash.dependencies.Input('slider-dim-2', 'value'),
    dash.dependencies.Input('slider-birth-date', 'value'),
    dash.dependencies.Input('slider-death-date', 'value'),
    dash.dependencies.Input('artist-dropdown', 'value'),
    dash.dependencies.Input('technique-dropdown', 'value')])
def get_predictions(dim_1,dim_2,birth_date,death_date,artist,technique):
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
           '1 Author Death':death_date,'Dim 1':dim_1,'Dim 2':dim_2,'Dim 3':dim_3,'Year':datetime.date.today().year,
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
    
  path = '.\\assets\\'
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
  df_map_author = pd.read_csv('.\\assets\\author_mapping_mean_encodings.csv')
  df_map_colours = pd.read_csv('.\\assets\\colours_mapping_mean_encodings.csv')

  df_inserted['1 Author_mean_encoded'] = df_inserted['1 Author'].map(df_map_author.set_index('1 Author').to_dict()['median'])
  df_inserted['Dominant Colour Name_mean_encoded'] = df['Final Price'].mean()
  #df_inserted['Dominant Colour Name_mean_encoded'] = df_inserted['Dominant Colour Name'].map(df_map_colours.set_index('Dominant Colour Name').to_dict()['median'])

  #Features to drop from feature importance
  with open(path+'feats_drop.npy', 'rb') as f:
      list_feats_drop = list(np.load(f,allow_pickle=True))  
  
  #Scaling 
  col_drop = ['1 Author','1 Author Birth','1 Author Death','Dim 1','Dim 2','Dim 3','Year','Technique','1 Author Birth Decade','1 Author Death Decade','Shape']
  scaler = joblib.load('.\\assets\\scaler.bin')

  #Ordered columns
  with open(path+'features_ordered.npy', 'rb') as f:
      list_reorder = list(np.load(f,allow_pickle=True))  

  df_final = df_inserted.drop(col_drop+list_feats_drop,axis=1)[list_reorder] #scaler.transform(df_inserted.drop(col_drop+list_feats_drop,axis=1)[list_reorder])

  #Make prediction
  #prediction = np.exp(model_xgb.predict(df_final))[0]
  prediction = np.exp(joblib_model.predict(df_final))[0]

  return 'Predicted auction price:\n{:.2f} €'.format(prediction)
