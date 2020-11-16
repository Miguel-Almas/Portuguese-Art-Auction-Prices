import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import dash_bootstrap_components as dbc

from app import app

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df_cml = pd.read_csv('C:\\Users\\miguel.almas\\OneDrive - insidemedia.net\\DataScience\\Portuguese-Art-Auction-Prices\\cleaned_data\\cleaned_df_cml.csv',index_col=0)

#Let's check a simple distribution plot of the Final Sale Price
# Create figure with secondary y-axis
df_cml['Date of Auction End'] = pd.to_datetime(df_cml['Date of Auction End']).dt.date

# ------------------------- Graph of sold artworks per auction number -----------------------------------
df_sale_record_artist = df_cml.groupby('Auction Number')['Sold'].agg(number_of_artworks='count',
                                                            number_of_sold_artworks='sum',
                                                            number_of_unsold_artworks=lambda x: x.count()-x.sum(),
                                                           sale_rate='mean').sort_values('number_of_artworks',ascending=False)
df_sale_record_artist = df_sale_record_artist.reset_index().sort_values('Auction Number',ascending=True)
df_sale_record_artist['Auction Number'] = df_sale_record_artist['Auction Number'].apply(lambda x: 'Auction Number '+str(x))

fig_auctions = go.Figure(data=[
    go.Bar(name='Sold Artworks', x=df_sale_record_artist['Auction Number'], y=df_sale_record_artist['number_of_sold_artworks']),
    go.Bar(name='Unsold Artworks', x=df_sale_record_artist['Auction Number'], y=df_sale_record_artist['number_of_unsold_artworks']),
])
# Change the bar mode
fig_auctions.update_layout(barmode='stack',title='Sold and unsold artworks per auction number')

layout  = html.Div(children=[
    html.H1(children='Modern and contemporary art',
        style={
            'textAlign': 'center'
                }
            ),
    html.H2(children='Sold in online portuguese auction houses',
        style={
            'textAlign': 'center'
                }
            ),

    html.Div(children='''
        This dashboard's goal is to allow exploration into the world of art, focusing on modern and contemporary portuguese artists.
    ''',
            style={
            'textAlign': 'center'
                }
            ),

    dcc.Graph(
        id='example-graph',
        figure=fig_auctions
    )
])