# -*- coding: utf-8 -*-
"""
Created on Sun May 16 00:30:06 2021

@author: andik
"""

import dash
import dash_table
import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv('FilterMethodsofKMeans.csv')

app = dash.Dash(__name__)

app.layout = dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in df.columns],
    data=df.to_dict('records'),
)


if __name__ == '__main__':
    app.run_server(debug=True)