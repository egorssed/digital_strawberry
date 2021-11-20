
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def healt_visualization(preds):
    colors = {"Healthy":px.colors.qualitative.Plotly[0], "Scab":px.colors.qualitative.Plotly[0], "Rust":px.colors.qualitative.Plotly[0], "Multiple diseases":px.colors.qualitative.Plotly[0]}
    if list.index(preds.tolist(), max(preds)) == 0:
        pred = "Healthy"
    if list.index(preds.tolist(), max(preds)) == 1:
        pred = "Scab"
    if list.index(preds.tolist(), max(preds)) == 2:
        pred = "Rust"
    if list.index(preds.tolist(), max(preds)) == 3:
        pred = "Multiple diseases"

    colors[pred] = px.colors.qualitative.Plotly[1]
    colors["Healthy"] = "seagreen"
    colors = [colors[val] for val in colors.keys()]
    fig = go.Figure(go.Bar(x=["Healthy", "Scab", "Rust", "Multiple diseases"], y=preds, marker=dict(color=colors)))
    fig.update_layout(height=400, width=800, title_text="Health Analysis", showlegend=False)
    pio.write_html(fig, file="static/plots/health.html", include_plotlyjs="cdn")
