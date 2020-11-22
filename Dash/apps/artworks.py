import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from PIL import Image
import dash_bootstrap_components as dbc
import pathlib

from app import app

df_cml = pd.read_csv('.\\assets\\data_cml_processed.csv',index_col=0)
df_cml = df_cml[df_cml['1 Author'] != 'no information']
df_cml['Final Price'] = np.exp(df_cml['Final Price'])

#------------------------ Create a WordCount Image to display in Dash---------------------------------------------

#Get Series with Artist Name Frequency
di = df_cml[df_cml['1 Author'] != 'no information']['1 Author'].value_counts()
#Define features that can be used to produce graphs
list_feats = ['1 Author','1 Author Birth Decade','Auction Number','Technique','Shape','Dominant Colour Name']

#Create layout
layout  = html.Div(children=[
    html.H1(children='Artworks',
        style={
            'textAlign': 'center'
                }
            ),

    html.Div(children='''
        Here exploration of artworks can be performed, with filtering available for specific artists or auctions.
    ''',
            style={
            'textAlign': 'center','margin-bottom':'20px',
                }
            ),
    dbc.Col(dbc.Row([
        dbc.Col([
            dbc.Row([
                dcc.Dropdown(id='artist-dropdown',
                            options=[{'label':i,'value':i} for i in di.index],
                            placeholder="Select one or more artists",
                            multi=True,
                            style={"width": "100%", 'margin-left':'20px', 'margin-right':'0px'},
                ),
            ]),
            dbc.Row([
                dbc.Col(
                	html.Div([
                        html.P("Number of Artists"),
                        html.H6(id='nbr_artists', className="number_artists")
                    ],id="artists",className="pretty_container",style={'Align': 'center',
                    }),width=6,xs=6,sm=6,md=6,lg=6,xl=6
                ),
                dbc.Col(
                	html.Div([
                        html.P("Number of Artworks"),
                        html.H6(id='nbr_artists_artworks', className="number_artworks")
                    ],id="artworks",className="pretty_container",style={'Align': 'center',
                    }),width=6,xs=6,sm=6,md=6,lg=6,xl=6
                ),
            ],style={'textAlign': 'center','margin-left':'25px','margin-top':'20px','margin-bottom':'20px'}),
            dbc.Row([
                dbc.Col(
                    html.Div(children='''
                        Artist and artwork information.
                        ''',
                    style={
                    'Align': 'center',
                    }),
                style={'margin-left':'20px'},width=6,xs=12,sm=12,md=6,lg=6,xl=6,)
            ]),
            dbc.Row([
                dbc.Col(
                    dash_table.DataTable(
                    id='table_artists',
                    data=[],
                    style_table={'height': '280px','width':'100%', 'overflowY': 'auto', 'margin-left':'20px','margin-right':'0px','overflowX': 'auto'}, #'280px'
                    style_cell={'fontSize':10,'height': 'auto','whiteSpace': 'normal'},#, 'font-family':'roboto'}
                    style_cell_conditional=[
                        {'if': {'column_id': 'Author'},
                            'width': '110px'},
                        {'if': {'column_id': 'Technique'},
                            'width': '135px'},
                        {'if': {'column_id': 'Dimensions'},
                            'width': '120px'},
                        {'if': {'column_id': 'Author Birth'},
                            'width': '80px'},
                        {'if': {'column_id': 'Author Death'},
                            'width': '100px'},
                        {'if': {'column_id': 'Price'},
                            'width': '40px'},
                        ]
                    ),
                style={'margin-bottom':'20px'}),#width=6,xs=12,sm=12,md=5,lg=5,xl=5,),
            ]),
        ],width=6,xs=11,sm=11,md=5,lg=5,xl=5,),
        #Create an image slideshow of the artists paintings
        dbc.Col([
            dbc.Row([
                html.Div([
                    html.Section(id="slideshow", children=[
                        html.Div(id="slideshow-container", children=[
                            html.Div(id="image"),
                            dcc.Interval(id='interval', interval=7000)
                        ])
                    ])

                ],style={'width':'90%'}),
            ],style={'padding': 0,'margin-left':0}),
            #Add information on the painting that is being displayed
            html.Div(id='artwork_artist',
                style={
                    'textAlign': 'left',
                    'margin-top':'25px',
                    'width':'90%',
                }),
            html.Div(id='artwork_title',
                style={
                    'textAlign': 'left',
                    'width':'90%',
                }),
            html.Div(id='artwork_technique',
                style={
                    'textAlign': 'left',
                    'width':'90%',
                }),
            html.Div(id='artwork_colour',
                style={
                    'textAlign': 'left',
                    'width':'90%',
                }),
            html.Div(id='artwork_price',
                style={
                    'textAlign': 'left',
                    'width':'90%',
                }),
        ],style={'margin-left': 40},width=6,xs=10,sm=10,md=6,lg=6,xl=6,),
    ],style={'padding': 0,'margin-bottom':20})),

    #Line breaker

    dbc.Col([
            dbc.Card(
                html.H2(children='Graphical Exploration',
                    style={
                        'textAlign': 'center',
                    },
                ),
                style={'padding': 20,},#'height':'70px'},#color="dark", inverse=False,
            ),
            #Breakdowns for furher filtering
            dbc.Row([
                dcc.Dropdown(id='col_to_analyse',
                    options=[{'label':i,'value':i} for i in list_feats],
                    placeholder="Select feature to analyse",
                    multi=False,
                    value = '1 Author',
                    style={"width": "97%",'height':'35px','margin-left':'20px', 'margin-right':'20px'},
                )],style={'padding':'20px'},
            ),

            #Rows for graphs
            dbc.Row([
                dbc.Col(
                    dcc.Graph(
                        id='graph_nbr_sold_unsold_artworks',
                    ),
                width=6,xs=12,sm=12,md=6,lg=6,xl=6,),
                dbc.Col(
                    dcc.Graph(
                        id='graph_boxplot_artists',
                    ),
                width=6,xs=12,sm=12,md=6,lg=6,xl=6,),   
            ]),
        ],style={'margin-top':20}),
])

######################### Callback #####################################
#Calback and functions for the artists paintings - get random row from selected artists and returns its index
@app.callback(dash.dependencies.Output('image', 'children'),
              dash.dependencies.Output('artwork_artist', 'children'),
              dash.dependencies.Output('artwork_title', 'children'),
              dash.dependencies.Output('artwork_technique', 'children'),
              dash.dependencies.Output('artwork_price', 'children'),
              dash.dependencies.Output('artwork_colour', 'children'),
              [dash.dependencies.Input('interval', 'n_intervals'),
              dash.dependencies.Input('artist-dropdown', 'value')])
def display_image(n,value):
    if value is None or len(value) == 0:
        #Get 1 random image from any author
        tmp_entry = df_cml.sample(1)
        tmp = pd.DataFrame(tmp_entry.apply(lambda x: str(x['Auction Number'])+'_'+str(x['Artwork Number']),axis=1))
        art_nbr = tmp[0].values[0]
        img = html.Img(src = app.get_asset_url('\\CML\\'+art_nbr+'.jpg'), style={'height':'60%'}) #'390px'
        #Also return some information on the painting title, technique, author, auction and sale price
        artwork_artist = str.title(str(tmp_entry['1 Author'].values[0]))
        artwork_title = str(tmp_entry['Title'].values[0])
        artwork_technique = str.capitalize(str(tmp_entry['Technique'].values[0]))
        artwork_colour = str.title(str(tmp_entry['Dominant Colour Name'].values[0]))
        if str.lower(str(tmp_entry['Sale Price'].values[0])) == 'not sold':
            artwork_price = 'Not sold'
        else:
            artwork_price = str(tmp_entry['Sale Price'].values[0])+' €'
    else:
        #Get 1 random image from 1 of selected authors
        filt = df_cml['1 Author'].isin(value)
        tmp_entry = df_cml[filt].sample(1)
        tmp = pd.DataFrame(tmp_entry.apply(lambda x: str(x['Auction Number'])+'_'+str(x['Artwork Number']),axis=1))
        art_nbr = tmp[0].values[0]
        img = html.Img(src = app.get_asset_url('\\CML\\'+art_nbr+'.jpg'), style={'height':'60%'})
        #Also return some information on the painting title, technique, author, auction and sale price
        artwork_artist = str.title(str(tmp_entry['1 Author'].values[0]))
        artwork_title = str(str(tmp_entry['Title'].values[0]))
        artwork_technique = str.capitalize(str(tmp_entry['Technique'].values[0]))
        artwork_colour = str.title(str(tmp_entry['Dominant Colour Name'].values[0]))
        if str.lower(str(tmp_entry['Sale Price'].values[0])) == 'not sold':
            artwork_price = 'Not sold'
        else:
            artwork_price = str(tmp_entry['Sale Price'].values[0])+' €'
    return img,artwork_artist,artwork_title,artwork_technique,artwork_price,artwork_colour

# Calculate number of selected artists
@app.callback(
    dash.dependencies.Output('nbr_artists', 'children'),
    [dash.dependencies.Input('artist-dropdown', 'value')])
def count_nbr_authors(value):
    if value is None or len(value) == 0:
        return df_cml['1 Author'].nunique()
    else:
        return df_cml[df_cml['1 Author'].isin(value)]['1 Author'].nunique()

# Calculate number of artworks by artists
@app.callback(
    dash.dependencies.Output('nbr_artists_artworks', 'children'),
    [dash.dependencies.Input('artist-dropdown', 'value')])
def count_nbr_authors_artworks(value):
    if value is None or len(value) == 0:
        return df_cml['1 Author'].shape[0]
    else:
        return df_cml[df_cml['1 Author'].isin(value)].shape[0]

#Create graph of sold and unsold artworks by artist
@app.callback(
    dash.dependencies.Output(component_id = 'graph_nbr_sold_unsold_artworks', component_property='figure'),
    [dash.dependencies.Input('artist-dropdown', 'value'),
    dash.dependencies.Input('col_to_analyse', 'value')])
def graph_nbr_sold_unsold_artworks(value,col):
    df_cml_tmp = df_cml.copy()
    if col == 'Auction Number':
        df_cml_tmp['Auction Number'] = df_cml_tmp['Auction Number'].apply(lambda x: 'Action '+str(x))

    if col == 'Technique':
        if value is not None and len(value) != 0:
            df_cml_tmp = df_cml_tmp[df_cml_tmp['1 Author'].isin(value)]

        cols_techniques = ['assinar', 'papel', 'datar', 'numerada', 'serigrafia', 'tecnica',
       'marcar', 'tela', 'misturar', 'defeito', 'pequeno', 'oleo', 'sinal',
       'europeu', 'uso', 'metal', 'decoracao', 'vidro', 'verso', 'acrilico',
       'madeira', 'escultura', 'identificar', 'português', 'gravura',
       'policromada', 'assinatura', 'aguarela', 'tinta', 'china', 'fabricar',
       'falta', 'cromado', 'italiano', 'portugue', 'material', 'azulejo',
       'platex', 'mancha', 'cristal', 'dourar', 'autor', 'pintado',
       'prateado', 'plastico', 'prova', 'hc', 'relevado', 'base']
        cols_targets = ['Sold','Final Price']
        cols = cols_techniques + cols_targets

        #Make subsets of sold and unsold artworks
        filt = df_cml_tmp[cols]['Sold'] == 1
        tmp_sold = df_cml_tmp[cols][filt]
        tmp_unsold = df_cml_tmp[cols][~filt]

        #Calculate sold and unsold artworks by technique
        series_sold = tmp_sold.drop(['Sold','Final Price'],axis=1).sum()
        series_unsold = tmp_unsold.drop(['Sold','Final Price'],axis=1).sum()

        #Drop column if series_sold + series_unsold still equals 0 artworks
        aux = pd.DataFrame(series_sold + series_unsold)[pd.DataFrame(series_sold + series_unsold)[0] <= 0].index
        series_sold = series_sold.drop(aux)
        series_unsold = series_unsold.drop(aux)

        #Make plot
        # Create figure with secondary y-axis
        fig_sold_unsold_artworks = make_subplots(specs=[[{"secondary_y": True}]])

        fig_sold_unsold_artworks.add_trace(go.Bar(name='Sold Artworks', x=series_sold.index, y=series_sold.values),secondary_y=False)
        fig_sold_unsold_artworks.add_trace(go.Bar(name='Unsold Artworks', x=series_sold.index, y=series_unsold.values),secondary_y=False)
        fig_sold_unsold_artworks.add_trace(go.Scatter(name='Sale Rate', x=series_sold.index, y=(series_sold.values/(series_sold+series_unsold).values)),secondary_y=True)

        # Change the bar mode
        fig_sold_unsold_artworks.update_layout(barmode='stack',title='Sold and unsold artworks by technique characteristic',hovermode="x unified",showlegend=False)
        # Set y-axes titles
        fig_sold_unsold_artworks.update_yaxes(title_text="Number artworks", secondary_y=False)
        fig_sold_unsold_artworks.update_yaxes(title_text="Sale rate", secondary_y=True)

    else:  
        if value is None or len(value) == 0:
            #Sold and unsold artwork by artist
            df_sale_record_artist = df_cml_tmp.groupby(col)['Sold'].agg(number_of_auctions='count',
                                                                number_of_sold_artworks='sum',
                                                                number_of_unsold_artworks=lambda x: x.count()-x.sum(),
                                                               sale_rate='mean').sort_values('number_of_auctions',ascending=False)
        else:
            #Sold and unsold artwork by artist
            df_sale_record_artist = df_cml_tmp[df_cml_tmp['1 Author'].isin(value)].groupby(col)['Sold'].agg(number_of_auctions='count',
                                                                number_of_sold_artworks='sum',
                                                                number_of_unsold_artworks=lambda x: x.count()-x.sum(),
                                                               sale_rate='mean').sort_values('number_of_auctions',ascending=False)        
        if col == 'Auction Number':
            df_sale_record_artist = df_sale_record_artist.sort_index(ascending=True)
        fig_sold_unsold_artworks = go.Figure(data=[
            go.Bar(name='Sold Artworks', x=df_sale_record_artist[:50].index, y=df_sale_record_artist[:50]['number_of_sold_artworks']),
            go.Bar(name='Unsold Artworks', x=df_sale_record_artist[:50].index, y=df_sale_record_artist[:50]['number_of_unsold_artworks'])
        ])
        # Change the bar mode
        fig_sold_unsold_artworks.update_layout(barmode='stack',title='Sold and unsold artworks by '+ col,hovermode="x unified",showlegend=False)

    return fig_sold_unsold_artworks

#Create boxplot
@app.callback(
    dash.dependencies.Output(component_id = 'graph_boxplot_artists', component_property='figure'),
    [dash.dependencies.Input('artist-dropdown', 'value'),
    dash.dependencies.Input('col_to_analyse', 'value')])
def graph_boxplot_artists(value,col):
    df_cml_tmp = df_cml.copy()
    df_cml_tmp = df_cml_tmp[df_cml_tmp['Final Price'] > 0]
    if col == 'Auction Number':
        df_cml_tmp['Auction Number'] = df_cml_tmp['Auction Number'].apply(lambda x: 'Action '+str(x))

    if col == 'Technique':
        if value is not None and len(value) != 0:
            df_cml_tmp = df_cml_tmp[df_cml_tmp['1 Author'].isin(value)]

        cols_techniques = ['assinar', 'papel', 'datar', 'numerada', 'serigrafia', 'tecnica',
       'marcar', 'tela', 'misturar', 'defeito', 'pequeno', 'oleo', 'sinal',
       'europeu', 'uso', 'metal', 'decoracao', 'vidro', 'verso', 'acrilico',
       'madeira', 'escultura', 'identificar', 'português', 'gravura',
       'policromada', 'assinatura', 'aguarela', 'tinta', 'china', 'fabricar',
       'falta', 'cromado', 'italiano', 'portugue', 'material', 'azulejo',
       'platex', 'mancha', 'cristal', 'dourar', 'autor', 'pintado',
       'prateado', 'plastico', 'prova', 'hc', 'relevado', 'base']

        fig_boxplot_sale_price = go.Figure()
        #obtain number of times the technique appears on the dataset to exclude from plot
        aux = df_cml_tmp[cols_techniques].sum()
        for i in aux[aux > 0].index:
            tmp = df_cml_tmp[df_cml_tmp[i] == 1]
            fig_boxplot_sale_price.add_trace(go.Box(name=i, y=tmp["Final Price"]))

        fig_boxplot_sale_price.update_layout(title='Boxplot of artwork price by technique characteristic',hovermode="x unified",showlegend=False)

    else:
        if value is None or len(value) == 0:
            #Sold and unsold artwork by artist
            df_sale_record_artist = df_cml_tmp.groupby(col)['Sold'].agg(number_of_auctions='count',
                                                                number_of_sold_artworks='sum',
                                                                number_of_unsold_artworks=lambda x: x.count()-x.sum(),
                                                               sale_rate='mean').sort_values('number_of_auctions',ascending=False)
        else:
            #Sold and unsold artwork by artist
            df_sale_record_artist = df_cml_tmp[df_cml_tmp['1 Author'].isin(value)].groupby(col)['Sold'].agg(number_of_auctions='count',
                                                                number_of_sold_artworks='sum',
                                                                number_of_unsold_artworks=lambda x: x.count()-x.sum(),
                                                               sale_rate='mean').sort_values('number_of_auctions',ascending=False) 
        if col == 'Auction Number':
            df_sale_record_artist = df_sale_record_artist.sort_index(ascending=True)       
        #Boxplot of sale prices by artwork
        #First, let's reoder the dataframe so that the same order of the previous graphs (from artist with most auctions to the one with the lowest)
        tmp = df_cml_tmp.copy()
        tmp['Temp'] = pd.CategoricalIndex(tmp[col], ordered=True, categories=df_sale_record_artist.index)
        df_cml_sorted = tmp.sort_values('Temp',ascending=True)
        #Create graph
        filt = (df_cml_sorted[col].isin(df_sale_record_artist[:50].index)) & (df_cml_sorted['Sold'] == 1)
        if col == 'Dominant Colour Name':
            fig_boxplot_sale_price = px.box(df_cml_sorted[filt], x='Dominant Colour Name', y="Final Price",title='Boxplot of final sale price for dominant colours associated with 0+ artworks',
                     color='Dominant Colour Name',color_discrete_sequence =list(df_sale_record_artist[df_sale_record_artist>0].index),points=False,boxmode="overlay")
        else:
            fig_boxplot_sale_price = px.box(df_cml_sorted[filt], x=col, y="Final Price",title='Final sale prices box plot') 
            fig_boxplot_sale_price.update_layout(hovermode="x unified",showlegend=False)

    return fig_boxplot_sale_price

#Data table Callback
@app.callback(
    [dash.dependencies.Output("table_artists", "data"), dash.dependencies.Output('table_artists', 'columns')],
    [dash.dependencies.Input('artist-dropdown', 'value')]
)
def table_artists(value):
    if value is None or len(value) == 0:
        tmp = df_cml
    else:
        tmp = df_cml[df_cml['1 Author'].isin(value)]
    
    tmp = tmp.rename(columns={'1 Author':'Author','Sale Price':'Price','1 Author Birth':'Author Birth','1 Author Death':'Author Death'})
    columns=[{"name": i, "id": i} for i in tmp[['Author', 'Technique', 'Dimensions',#'Title',
                                                'Author Birth', 'Author Death',
                                                'Price']].columns]
    data=tmp.to_dict('records')

    return data, columns