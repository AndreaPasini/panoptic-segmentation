import os
from skimage import io
import plotly.express as px
import matplotlib.pyplot as plt
import pyximport
pyximport.install(language_level=3)
import dash_interactive_graphviz
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from sims.sgs import read_freqgraphs

from main_sims2 import get_maximal_itemsets
from sims.graph_utils import json_to_graphviz
from sims.sims_config import SImS_config

if __name__ == '__main__':
    config = SImS_config('COCO')
    fgraphs = read_freqgraphs(config)
    maximal_fgraphs = get_maximal_itemsets(fgraphs)

    # components = [html.Div(children=dash_interactive_graphviz.DashInteractiveGraphviz(id=f"graph{i}",
    #                                     dot_source=json_to_graphviz(g['g']).source),
    #                         style = {'width': '200px', 'float': 'left', 'display': 'block'})
    #                 for i,g in enumerate(maximal_fgraphs)]
    app = dash.Dash(__name__)



    #fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig = px.imshow(io.imread(os.path.join(config.SGS_dir,'charts/places/mountain_outside_92.png')),width=400)
    fig2 = px.imshow(io.imread(os.path.join(config.SGS_dir,'charts/places/mountain_outside_92.png')),width=400)
    #fig = plt.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

    app.layout = html.Div([
        html.H1("SImS, SGS exploration.", style={'text-align' : 'center'}),
        dcc.Dropdown(id='choice', options=[{'label':'a', 'value':1},{'label':'b', 'value':2}],
                    value=1, multi=False),
        html.Div(id='output_sgs', children=[]),
        #html.Div(id='components', children=components[:3])
        dcc.Graph(id='chart', figure=fig, animate=False, style={'width': '600px', 'float': 'left', 'display': 'block'}),
        dcc.Graph(id='chart2', figure=fig2, animate=False, style={'width': '600px', 'float': 'left', 'display': 'block'}),
        html.Img(id='chart3', src=os.path.abspath(os.path.join(config.SGS_dir,'charts/places/mountain_outside_92.png')), style={'width': '600px', 'float': 'left', 'display': 'block'})

        #dcc.Graph(id='chart')
        #html.Iframe(id='chart', src="")
    ])

    # @app.callback(
    #     [Output(component_id='output_sgs', component_property='children'),
    #      Output(component_id='chart', component_property='src')],
    #     [Input(component_id='choice', component_property='value')]
    # )
    # def update_graph(option):
    #     print(option)
    #     #read_pdf = webbrowser.open_new(os.path.join(config.SGS_dir,'charts/maximal/g99_s_1084.pdf'))
    #     return (f"You selected {option}",os.path.join(config.SGS_dir,'charts/maximal/g99_s_1084.pdf'))
    #


    app.run_server(debug=True)