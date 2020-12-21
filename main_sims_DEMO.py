import os
from skimage import io
import plotly.express as px
import matplotlib.pyplot as plt
import pyximport
pyximport.install(language_level=3)
#import dash_interactive_graphviz
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from sims.sgs import load_SGS

from main_sims_places_test import get_maximal_itemsets
from sims.graph_utils import json_to_graphviz
from sims.sims_config import SImS_config


def create_figure(path, name):
    fig = px.imshow(io.imread(path), width=300, )
    fig.update_layout({'hovermode': False, 'margin' : dict(l=20, r=20, t=20, b=20)})

    graph = dcc.Graph(id=name, figure=fig, animate=False,
              style={'width': '300px', 'float': 'left', 'display': 'block'},
              config={'displayModeBar': False, 'scrollZoom': True})
    return graph

def create_figures(n):
    files = os.listdir(os.path.join(config.SGS_dir, 'charts/sgs_eprune_nprune_gspan_05/'))
    figures = []
    i = 0
    for file in files:
        if i >= n: break
        if file.endswith('.jpg'):
            figures.append(create_figure(os.path.join(config.SGS_dir, 'charts/sgs_eprune_nprune_gspan_05/', file),
                                         f'chart{i}'))
            i += 1
    return figures


if __name__ == '__main__':
    # import plotly.graph_objs as go
    # layout = go.Layout(
    #     margin=go.layout.Margin(
    #         l=0,  # left margin
    #         r=0,  # right margin
    #         b=0,  # bottom margin
    #         t=0  # top margin
    #     )
    # )



    config = SImS_config('COCO_subset2')
    config.SGS_params['minsup']=0.05
    fgraphs = load_SGS(config)
    maximal_fgraphs = get_maximal_itemsets(fgraphs)

    # components = [html.Div(children=dash_interactive_graphviz.DashInteractiveGraphviz(id=f"graph{i}",
    #                                     dot_source=json_to_graphviz(g['g']).source),
    #                         style = {'width': '200px', 'float': 'left', 'display': 'block'})
    #                 for i,g in enumerate(maximal_fgraphs)]
    app = dash.Dash(__name__)



    #fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    #fig = plt.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

    # To remove toolbar options:
    # config={
    #     'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
    # }

    elements = [
        html.H1("SImS, SGS generation.", style={'text-align': 'center'}),
        html.Div([html.Caption('Select COCO subset:', style={'float':'left', 'width':'300px'}), html.Textarea()], style={'width':'800px'}),
        html.H1("SImS, SGS exploration.", style={'text-align' : 'center'}),
        dcc.Dropdown(id='choice', options=[{'label':'5', 'value':5},{'label':'10', 'value':10}],
                    value=5, multi=False),
        html.Div(create_figures(5), id='figuresDiv')
    ]

    app.layout = html.Div(elements)

    @app.callback(
        [Output(component_id='figuresDiv', component_property='children')],
        [Input(component_id='choice', component_property='value')]
    )
    def update_graph(option):
        return (create_figures(option),)


    app.run_server(debug=True)