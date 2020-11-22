import dash
import dash_bootstrap_components as dbc

# bootstrap theme
# https://bootswatch.com/lux/
external_stylesheets = [dbc.themes.LUX]
#external_stylesheets=[dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, meta_tags=[
                #{"name": "viewport", "content": "width=device-width, initial-scale=1"},
                {"name": "HandheldFriendly", "content": "true"},
                {"name": "viewport", "content": "width=device-width,initial-scale=1.0,minimum-scale=1.0, maximum-scale=1.0, user-scalable=yes"},
            ], external_stylesheets=external_stylesheets)

server = app.server
app.config.suppress_callback_exceptions = True