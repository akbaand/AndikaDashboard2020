import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.io as io
import dash_bootstrap_components as dbc
import dash_table

external_stylesheets = [dbc.themes.CERULEAN]

#Prepare the Data
df_EDA_raw = pd.read_csv('energy_data_raw.csv')
df_EDA_Zscore = pd.read_csv('energy_data_Zscore.csv')
df_EDA_IQR = pd.read_csv('energy_data_IQR.csv')

#Prepare Data for Feature Selection
df_filter = pd.read_csv('FilterMethodsofKMeans.csv')
df_wrapper = pd.read_csv('WrapperMethodsRFE.csv')
df_ensemble = pd.read_csv('EnsembleMethod.csv')


# Prepare the data
x = np.linspace(1, 12)
y2017=x;
y2018=-2*x;
y2019=x*x; 

# Plot the data
plt.plot(x, y2017, label='2017')
plt.savefig('assets/graph2017.png')
plt.clf()
plt.plot(x, y2018, label='2018')
plt.savefig('assets/graph2018.png')
plt.clf()
plt.plot(x, y2019, label='2019')
plt.savefig('assets/graph2019.png')

fig1=px.scatter(x=x,y=y2017)
fig2=px.scatter(x=x,y=y2018)
fig3=px.scatter(x=x,y=y2019)

#Errors for Linear Regression
cardMAE_LR = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("MAE", className="card-title"),
            html.H6("Mean Absolute Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("17.54")]),
            html.P(
                "Linear Regression",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardMSE_LR = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("MSE", className="card-title"),
            html.H6("Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("554.59")]),
            html.P(
                "Linear Regression",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardRMSE_LR = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("RMSE", className="card-title"),
            html.H6("Root Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("23.54")]),
            html.P(
                "Linear Regression",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardcvRMSE_LR = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("cvRMSE", className="card-title"),
            html.H6("Coeff. of Var Root Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("0.11")]),
            html.P(
                "Linear Regression",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

#Errors for SVR

cardMAE_SVR = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("MAE", className="card-title"),
            html.H6("Mean Absolute Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("10.08")]),
            html.P(
                "Support Vector Regressor",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardMSE_SVR = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("MSE", className="card-title"),
            html.H6("Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("247.77")]),
            html.P(
                "Support Vector Regressor",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardRMSE_SVR = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("RMSE", className="card-title"),
            html.H6("Root Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("15.74")]),
            html.P(
                "Support Vector Regressor",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardcvRMSE_SVR = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("cvRMSE", className="card-title"),
            html.H6("Coeff. of Var Root Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("0.07")]),
            html.P(
                "Support Vector Regressor",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

#Errors for DT

cardMAE_DT = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("MAE", className="card-title"),
            html.H6("Mean Absolute Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("10.84")]),
            html.P(
                "Decision Tree Regressor",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardMSE_DT = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("MSE", className="card-title"),
            html.H6("Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("316.16")]),
            html.P(
                "Decision Tree Regressor",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardRMSE_DT = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("RMSE", className="card-title"),
            html.H6("Root Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("17.78")]),
            html.P(
                "Decision Tree Regressor",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardcvRMSE_DT = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("cvRMSE", className="card-title"),
            html.H6("Coeff. of Var Root Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("0.08")]),
            html.P(
                "Decision Tree Regressor",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

#Errors for NN

cardMAE_NN = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("MAE", className="card-title"),
            html.H6("Mean Absolute Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("14.64")]),
            html.P(
                "Neural Networks",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardMSE_NN = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("MSE", className="card-title"),
            html.H6("Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("474.10")]),
            html.P(
                "Neural Networks",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardRMSE_NN = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("RMSE", className="card-title"),
            html.H6("Root Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("21.77")]),
            html.P(
                "Neural Networks",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardcvRMSE_NN = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("cvRMSE", className="card-title"),
            html.H6("Coeff. of Var Root Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("0.10")]),
            html.P(
                "Neural Networks",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

#Errors for GB

cardMAE_GB = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("MAE", className="card-title"),
            html.H6("Mean Absolute Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("8.27")]),
            html.P(
                "Gradient Boosting",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardMSE_GB = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("MSE", className="card-title"),
            html.H6("Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("173.37")]),
            html.P(
                "Gradient Boosting",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardRMSE_GB = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("RMSE", className="card-title"),
            html.H6("Root Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("13.16")]),
            html.P(
                "Gradient Boosting",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardcvRMSE_GB = dbc.Card(   

   [
    dbc.CardBody(
        [
            html.H4("cvRMSE", className="card-title"),
            html.H6("Coeff. of Var Root Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("0.063")]),
            html.P(
                "Gradient Boosting",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

#Errors for XGB

cardMAE_XGB = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("MAE", className="card-title"),
            html.H6("Mean Absolute Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("8.04")]),
            html.P(
                "Extreme Gradient Boosting",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardMSE_XGB = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("MSE", className="card-title"),
            html.H6("Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("171.69")]),
            html.P(
                "Extreme Gradient Boosting",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardRMSE_XGB = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("RMSE", className="card-title"),
            html.H6("Root Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("13.10")]),
            html.P(
                "Extreme Gradient Boosting",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardcvRMSE_XGB = dbc.Card(   

   [
    dbc.CardBody(
        [
            html.H4("cvRMSE", className="card-title"),
            html.H6("Coeff. of Var Root Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("0.063")]),
            html.P(
                "Extreme Gradient Boosting",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),
    
#Errors for BS

cardMAE_BS = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("MAE", className="card-title"),
            html.H6("Mean Absolute Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("8.34")]),
            html.P(
                "BootStrapping",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardMSE_BS = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("MSE", className="card-title"),
            html.H6("Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("185.82")]),
            html.P(
                "BootStrapping",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardRMSE_BS = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("RMSE", className="card-title"),
            html.H6("Root Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("13.63")]),
            html.P(
                "BootStrapping",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardcvRMSE_BS = dbc.Card(   

   [
    dbc.CardBody(
        [
            html.H4("cvRMSE", className="card-title"),
            html.H6("Coeff. of Var Root Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("0.065")]),
            html.P(
                "BootStrapping",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),
  
#Errors for RF

cardMAE_RF = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("MAE", className="card-title"),
            html.H6("Mean Absolute Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("7.93")]),
            html.P(
                "Random Forest Regressor",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardMSE_RF = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("MSE", className="card-title"),
            html.H6("Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("164.66")]),
            html.P(
                "Random Forest Regressor",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardRMSE_RF = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("RMSE", className="card-title"),
            html.H6("Root Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("12.83")]),
            html.P(
                "Random Forest Regressor",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardcvRMSE_RF = dbc.Card(   

   [
    dbc.CardBody(
        [
            html.H4("cvRMSE", className="card-title"),
            html.H6("Coeff. of Var Root Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("0.061")]),
            html.P(
                "Random Forest Regressor",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),  

#Errors for RFI

cardMAE_RFI = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("MAE", className="card-title"),
            html.H6("Mean Absolute Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("8.33")]),
            html.P(
                "Random Forest Unformized",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardMSE_RFI = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("MSE", className="card-title"),
            html.H6("Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("178.91")]),
            html.P(
                "Random Forest Unformized",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardRMSE_RFI = dbc.Card(
   [
    dbc.CardBody(
        [
            html.H4("RMSE", className="card-title"),
            html.H6("Root Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("13.37")]),
            html.P(
                "Random Forest Unformized",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),

cardcvRMSE_RFI = dbc.Card(   

   [
    dbc.CardBody(
        [
            html.H4("cvRMSE", className="card-title"),
            html.H6("Coeff. of Var Root Mean Squared Error", className="card-subtitle"),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem("0.064")]),
            html.P(
                "Random Forest Unformized",
                className="card-text",style={"color":"white"}
            ),
        ]),
    ],
   color="dark",
   inverse=False,
   outline=False,
   style={"width": "18rem",
          "height":"13rem"},
),  

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Img(src=app.get_asset_url('IST_logo.png'),style={'height':'10%', 'width':'10%'}),
    html.H2('Energy Services Dashboard'),
    html.H2('by Andika Akbar Hermawan'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='EDA', value='tab-1'),
        dcc.Tab(label='Clustering', value='tab-2'),
        dcc.Tab(label='Feature', value='tab-3'),
        dcc.Tab(label='Regression', value='tab-4'),
        
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
             

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Exploratory Data Analysis'),
            dcc.RadioItems(
        id='radio',
        options=[
            {'label': 'Raw Data Before Cleaning', 'value': 2017},
            {'label': 'Data with Removed Outliers (ZScore)', 'value': 2018},
            {'label': 'Data with Removed Outliers (IQR)', 'value': 2019}
        ], 
        value=2017
        ),
        html.Div(id='graphyear_png'),
                    ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Clustering'),
            dcc.Dropdown( 
        id='dropdown',
        options=[
            {'label': '2017', 'value': 2017},
            {'label': '2018', 'value': 2018},
            {'label': '2019', 'value': 2019}
        ], 
        value=2017
        ),
        html.Div(id='graphyear_html'),
                    
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Feature selection'),
            dcc.Dropdown(
                id='dropdown',
                options=[
                    {'label':'Filter Methods of KMeans','value':1},
                    {'label':'Wrapper Methods RFE','value':2},
                    {'label':'Ensemble Methods','value':3}
                    ],
                value=1
                ),
            html.Div(id='featuretable_png')
        ])
    
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Regression'),
            dcc.Dropdown( 
        id='dropdown',
        options=[
            {'label': 'Linear Progression', 'value': 3},
            {'label': 'Support Vector Machine', 'value': 4},
            {'label': 'Decision Tree', 'value': 5},
            {'label': 'Neural Networks','value': 6},
            {'label': 'Gradient Boosting','value': 7},
            {'label': 'Extreme Gradient Boosting','value': 8},
            {'label': 'Bootstraping','value': 9},
            {'label': 'Random Forest','value': 10},
            {'label': 'Random Forest Uninformized', 'value':11},
        ], 
        value=3
        ),
        html.Div(id='regressionmethod_png'),
                    
        ])

@app.callback(Output('graphyear_png', 'children'), 
              Input('radio', 'value'))

def render_figure_png(radio_year):
    
    if radio_year == 2017:
        return html.Div([dcc.Graph(
            id='raw-data',
            figure={
                'data':[
                    {'x':df_EDA_raw.Datetime,'y':df_EDA_raw.Power_kW,'type':'scatter','name':'EDA Raw Data'}]})])
    elif radio_year == 2018:
        return html.Div([dcc.Graph(
            id='raw-data',
            figure={
                'data':[
                    {'x':df_EDA_Zscore.Datetime,'y':df_EDA_Zscore.Power_kW,'type':'scatter','name':'EDA with Removed Outliers ZScore'}]})])
    elif radio_year == 2019:
        return html.Div([dcc.Graph(
            id='raw-data',
            figure={
                'data':[
                    {'x':df_EDA_IQR.Datetime,'y':df_EDA_IQR.Power_kW,'type':'scatter','name':'EDA with Removed Outliers IQR'}]})])
    
@app.callback(Output('graphyear_html', 'children'), 
              Input('dropdown', 'value'))

def render_figure_html(dropdown_year):
    
    if dropdown_year == 2017:
        return html.Div([dcc.Graph(figure=fig1),])
    elif dropdown_year == 2018:
        return html.Div([dcc.Graph(figure=fig2),])
    elif dropdown_year == 2019:
        return html.Div([dcc.Graph(figure=fig3),])

@app.callback(Output('featuretable_png','children'),
              Input('dropdown','value'))

def render_feature_png(dropdown_year):
    if dropdown_year == 1:
        return html.Div([
            dash_table.DataTable(
                style_cell={
                    'whiteSpace':'normal',
                    'height':'auto'},
                id='table',
                columns=[{"name":i,"id":i} for i in df_filter.columns],
                data=df_filter.to_dict('records'),
                ),
            html.H2('Filter Selection with KMeans indicate that the best K value are coming from Solar Radiation and Power-1')
            ])
    elif dropdown_year == 2:
        return html.Div([
            dash_table.DataTable(
                id='table',
                columns=[{"name":i,"id":i} for i in df_wrapper.columns],
                data=df_wrapper.to_dict('records'),
                ),
            html.H2('Wrapper Methods with 2 Features, Best Values Holiday and Hour'),
            html.H2('Wrapper Methods with 3 Features, Best Values Holiday, Power-1, Hour')
            ])
    elif dropdown_year == 3:
        return html.Div([
            dash_table.DataTable(
                id='table',
                columns=[{"name":i,"id":i} for i in df_ensemble.columns],
                data=df_ensemble.to_dict('records'),
                ),
            html.H2('Ensemble Methods show that the Best Values are Power-1 and Hour')
            ])
    
@app.callback(Output('regressionmethod_png','children'),
              Input('dropdown','value'))


def render_regression_png(dropdown_year):
    if dropdown_year == 3:
        return html.Div([
            dbc.Row([
                dbc.Col(cardMAE_LR,width=3),
                dbc.Col(cardMSE_LR,width=3),
                dbc.Col(cardRMSE_LR,width=3),
                dbc.Col(cardcvRMSE_LR,width=3)
                ]),
            html.H3('Linear Progression Test vs Prediction'),
                html.Img(src='assets/LinearProgression_yTestPred.png',
                     style={
                         'height':'40%',
                         'width':'40%',
                         'display':'inline-block'}),
            html.H3('Linear Progression Scatter Plot'),
                    #style={'display':'inline-block'}),
                html.Img(src='assets/LinearProgression.png',
                     style={
                         'height':'40%',
                         'width':'40%',                
                         'display':'inline-block'}),
        
            ],style={
                'textAlign':'center'})
    elif dropdown_year == 4:
        return html.Div([
            dbc.Row([
                dbc.Col(cardMAE_SVR,width=3),
                dbc.Col(cardMSE_SVR,width=3),
                dbc.Col(cardRMSE_SVR,width=3),
                dbc.Col(cardcvRMSE_SVR,width=3)
                ]),
             html.H3('Support Vector Machine Test vs Prediction'),
                    #style={'display':'inline-block'}),
                html.Img(src='assets/SVR_yTestPred.png',
                     style={
                         'height':'40%',
                         'width':'40%',
                         'display':'inline-block'}),
            html.H3('Support Vector Machine Scatter Plot'),
                    #style={'display':'inline-block'}),
                html.Img(src='assets/SVR.png',
                     style={
                         'height':'40%',
                         'width':'40%',
                         'display':'inline-block'}),
            ],style={
                'textAlign':'center'})
    elif dropdown_year == 5:
        return html.Div([
            dbc.Row([
                dbc.Col(cardMAE_DT,width=3),
                dbc.Col(cardMSE_DT,width=3),
                dbc.Col(cardRMSE_DT,width=3),
                dbc.Col(cardcvRMSE_DT,width=3)
                ]),
             html.H3('Decision Tree Test vs Prediction'),
                    #style={'display':'inline-block'}),
                html.Img(src='assets/DecisionTree_yTestPred.png',
                     style={
                         'height':'40%',
                         'width':'40%',
                         'display':'inline-block'}),
            html.H3('Decision Tree Scatter Plot'),
                    #style={'display':'inline-block'}),
                html.Img(src='assets/DecisionTree.png',
                     style={
                         'height':'40%',
                         'width':'40%',
                         'display':'inline-block'}),
            ],style={
                'textAlign':'center'})
    elif dropdown_year == 6:
        return html.Div([
            dbc.Row([
                dbc.Col(cardMAE_NN,width=3),
                dbc.Col(cardMSE_NN,width=3),
                dbc.Col(cardRMSE_NN,width=3),
                dbc.Col(cardcvRMSE_NN,width=3)
                ]),
             html.H3('Neural Networks Test vs Prediction'),
                    #style={'display':'inline-block'}),
                html.Img(src='assets/NeuralNetworks_yTestPred.png',
                     style={
                         'height':'40%',
                         'width':'40%',
                         'display':'inline-block'}),
            html.H3('Neural Networks Scatter Plot'),
                    #style={'display':'inline-block'}),
                html.Img(src='assets/NeuralNetworks.png',
                     style={
                         'height':'40%',
                         'width':'40%',
                         'display':'inline-block'}),
            ],style={
                'textAlign':'center'})
    elif dropdown_year == 7:
        return html.Div([
            dbc.Row([
                dbc.Col(cardMAE_GB,width=3),
                dbc.Col(cardMSE_GB,width=3),
                dbc.Col(cardRMSE_GB,width=3),
                dbc.Col(cardcvRMSE_GB,width=3)
                ]),
            html.H3('Gradient Boosting Test vs Prediction'),
                    #style={'display':'inline-block'}),
                html.Img(src='assets/GradientBoosting_yTestPred.png',
                     style={
                         'height':'40%',
                         'width':'40%',
                         'display':'inline-block'}),
            html.H3('Gradient Boosting Scatter Plot'),
                    #style={'display':'inline-block'}),
                html.Img(src='assets/GradientBoosting.png',
                     style={
                         'height':'40%',
                         'width':'40%',
                         'display':'inline-block'}),
            ],style={
                'textAlign':'center'})
    elif dropdown_year == 8:
        return html.Div([
             dbc.Row([
                dbc.Col(cardMAE_XGB,width=3),
                dbc.Col(cardMSE_XGB,width=3),
                dbc.Col(cardRMSE_XGB,width=3),
                dbc.Col(cardcvRMSE_XGB,width=3)
                ]),
            html.H3('Extreme Gradient Boosting Test vs Prediction'),
                    #style={'display':'inline-block'}),
                html.Img(src='assets/XtremeGradBoost_yTestPred.png',
                     style={
                         'height':'40%',
                         'width':'40%',
                         'display':'inline-block'}),
            html.H3('Extreme Gradient Boosting Scatter Plot'),
                    #style={'display':'inline-block'}),
                html.Img(src='assets/XtremeGradBoost.png',
                     style={
                         'height':'40%',
                         'width':'40%',
                         'display':'inline-block'}),
            ],style={
                'textAlign':'center'})
    elif dropdown_year == 9:
        return html.Div([
            dbc.Row([
                dbc.Col(cardMAE_BS,width=3),
                dbc.Col(cardMSE_BS,width=3),
                dbc.Col(cardRMSE_BS,width=3),
                dbc.Col(cardcvRMSE_BS,width=3)
                ]),
            html.H3('Bootstrapping Test vs Prediction'),
                    #style={'display':'inline-block'}),
                html.Img(src='assets/Bootstraping_yTestPred.png',
                     style={
                         'height':'40%',
                         'width':'40%',
                         'display':'inline-block'}),
            html.H3('Bootstrapping Scatter Plot'),
                    #style={'display':'inline-block'}),
                html.Img(src='assets/Bootstraping.png',
                     style={
                         'height':'40%',
                         'width':'40%',
                         'display':'inline-block'}),
            ],style={
                'textAlign':'center'})
    elif dropdown_year == 10:
        return html.Div([
            dbc.Row([
                dbc.Col(cardMAE_RF,width=3),
                dbc.Col(cardMSE_RF,width=3),
                dbc.Col(cardRMSE_RF,width=3),
                dbc.Col(cardcvRMSE_RF,width=3)
                ]),
            html.H3('Random Forest Test vs Prediction'),
                    #style={'display':'inline-block'}),
                html.Img(src='assets/RandomForest_yTestPred.png',
                     style={
                         'height':'40%',
                         'width':'40%',
                         'display':'inline-block'}),
            html.H3('Random Forest Scatter Plot'),
                    #style={'display':'inline-block'}),
                html.Img(src='assets/RandomForest.png',
                     style={
                         'height':'40%',
                         'width':'40%',
                         'display':'inline-block'}),
            ],style={
                'textAlign':'center'})
    elif dropdown_year == 11:
        return html.Div([
            dbc.Row([
                dbc.Col(cardMAE_RFI,width=3),
                dbc.Col(cardMSE_RFI,width=3),
                dbc.Col(cardRMSE_RFI,width=3),
                dbc.Col(cardcvRMSE_RFI,width=3)
                ]),
            html.H3('Random Forest Unformized Test vs Prediction'),
                    #style={'display':'inline-block'}),
                html.Img(src='assets/RandomForestUni_yTestPred.png',
                     style={
                         'height':'40%',
                         'width':'40%',
                         'display':'inline-block'}),
            html.H3('Random Forest Unformized Scatter Plot'),
                    #style={'display':'inline-block'}),
                html.Img(src='assets/RandomForestUni.png',
                     style={
                         'height':'40%',
                         'width':'40%',
                         'display':'inline-block'}),
            ],style={
                'textAlign':'center'})  
    
    

if __name__ == '__main__':
    app.run_server(debug=True)
