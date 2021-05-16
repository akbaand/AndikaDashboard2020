# -*- coding: utf-8 -*-
"""
Created on Fri May 14 16:54:27 2021

@author: Andika Akbar Hermawan
"""

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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    
    html.Img(src=app.get_asset_url('IST_Logo.png'),style={'height':'10%', 'width':'10%'}),
    html.H2('IST Central Building Energy Forcast'),
    html.H3('by Andika Akbar Hermawan12'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='EDA', value='tab-1'),
        dcc.Tab(label='Clustering', value='tab-2'),
        dcc.Tab(label='Feature', value='tab-3'),
        dcc.Tab(label='Regression', value='tab-4'),
        
    ]),
    html.Div(id='tabs-content')#ID=dinamain masing2 tabs nya, bisa children or tabs content 
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
            {'label': 'Raw Data', 'value': 0},
            {'label': 'Removing Outliers with Z-Score', 'value': 1},
            {'label': 'Removing Outliers with IQR Method', 'value': 2}
        ], 
        value=0
        ),
        html.Div(id='EDA_png'),
                    ])
    elif tab == 'tab-2':
        return [
            html.Div([
            html.H3('Clustering'),
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
        html.Div(id='Clsutering_png'),
                    
        ])]
    
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Feature selection'),
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
        html.Div(id='Regression_png'),
                    
        ])
@app.callback(Output('EDA_png', 'children'), 
              Input('radio', 'value'))

def render_figure_png(radio_year):
    
    if radio_year == 0:
        return html.Div([html.Img(src='assets/EDA_Raw_Data1.png'),])
    elif radio_year == 1:
        return html.Div([html.Img(src=app.get_asset_url('EDA_Remove_ZScore1.png')),])
    elif radio_year == 2:
        return html.Div([html.Img(src=app.get_asset_url('EDA_Remove_IQR1.png')),])

    
@app.callback(Output('Regression_png', 'children'), 
              Input('dropdown', 'value'))

def render_figure_html(dropdown_year):
    
    if dropdown_year == 3:
        return html.Div([
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
            html.H3('Bootstrapping Test vs Prediction'),
                    #style={'display':'inline-block'}),
                html.Img(src='assets/Bootstrapping_yTestPred.png',
                     style={
                         'height':'40%',
                         'width':'40%',
                         'display':'inline-block'}),
            html.H3('Bootstrapping Scatter Plot'),
                    #style={'display':'inline-block'}),
                html.Img(src='assets/Bootstrapping.png',
                     style={
                         'height':'40%',
                         'width':'40%',
                         'display':'inline-block'}),
            ],style={
                'textAlign':'center'})
    elif dropdown_year == 10:
        return html.Div([
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
            html.H3('Random Forest Uninformized Test vs Prediction'),
                    #style={'display':'inline-block'}),
                html.Img(src='assets/RandomForestUni_yTestPred.png',
                     style={
                         'height':'40%',
                         'width':'40%',
                         'display':'inline-block'}),
            html.H3('Random Forest Scatter Plot'),
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
