import sys
sys.path.append('ext/Mask-RCNN-leekunhee/')

from mrcnn import visualize
import matplotlib.pyplot as plt

import numpy as np

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def health_visualization(preds):
    # prediction order = 'healthy', 'multiple_diseases', 'rust', 'scab'
    preds = np.array([preds[0], preds[2], preds[1] + preds[3]])
    classes = ["Здоровое", "Ржавчина", "Множественные заболевания"]
    colors = {classes[0]:px.colors.qualitative.Plotly[0],
              classes[1]:px.colors.qualitative.Plotly[0],
              classes[2]:px.colors.qualitative.Plotly[0]
    }
    print(preds)
    pred = classes[0]
    if list.index(preds.tolist(), max(preds)) == 0:
        pred = classes[0]
    if list.index(preds.tolist(), max(preds)) == 1:
        pred = classes[1]
    if list.index(preds.tolist(), max(preds)) == 2:
        pred = classes[2]

    colors[pred] = px.colors.qualitative.Plotly[1]
    colors["Здоровое"] = "seagreen"
    colors["Множественные заболевания"] = "red"
    colors = [colors[val] for val in colors.keys()]
    fig = go.Figure(go.Bar(x=classes, y=preds, marker=dict(color=colors)))
    fig.update_layout(height=400, width=800, title_text="Уверенность модели", title_x=0, showlegend=False)
    pio.write_html(fig, file="static/plots/health.html", include_plotlyjs="cdn")


def phase_visualization(preds):
    # Original order ['Berries', 'Flowers', 'Sapling']
    preds = preds[[2, 1, 0]]
    classes = ['Росток', 'Цветение', 'Ягоды']
    colors = {classes[0]:px.colors.qualitative.Plotly[0],
              classes[1]:px.colors.qualitative.Plotly[0],
              classes[2]:px.colors.qualitative.Plotly[0]
    }
    pred = classes[0]
    if list.index(preds.tolist(), max(preds)) == 0:
        pred = classes[0]
    if list.index(preds.tolist(), max(preds)) == 1:
        pred = classes[1]
    if list.index(preds.tolist(), max(preds)) == 2:
        pred = classes[2]

    colors[pred] = px.colors.qualitative.Plotly[1]
    colors["Росток"] = "seagreen"
    colors["Цветение"] = "yellow"
    colors["Ягоды"] = "red"
    colors = [colors[val] for val in colors.keys()]
    fig = go.Figure(go.Bar(x=classes, y=preds, marker=dict(color=colors)))
    fig.update_layout(height=400, width=800, title_text="Уверенность модели", title_x=0, showlegend=False)
    pio.write_html(fig, file="static/plots/phase.html", include_plotlyjs="cdn")


def seg_visualization(img, preds):
    def get_ax(rows=1, cols=1, size=16):
        """Return a Matplotlib Axes array to be used in
        all visualizations in the notebook. Provide a
        central point to control graph sizes.

        Adjust the size attribute to control how big to render images
        """
        _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
        return ax

    ax = get_ax(1)
    r = preds[0]
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                ['background', 'strawberry'], r['scores'], ax=ax,
                                title="Predictions")