import panel as pn
import numpy as np

# import pandas as pd

from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, HoverTool, CustomJS
# from bokeh.layouts import layout

from json import load
# import matplotlib
# import matplotlib.pyplot as plt
#
# from skimage import exposure
#
# import torch
# from torchvision import transforms
# import models
#
# import os
# import gc

# matplotlib.use('agg')

dataset = "candels_0"
emb = "UMAP"
tess = 1000
query_num = 50
seed = 0
qs = "LeastConfidence"

dir_path = "/media/gs15096/2F21743F2CB8AE68/PhD/Deep-Active-Learning"

metrics_path = f"{dir_path}/metrics/{dataset}/{emb}/{tess}/{query_num}/{qs}_any_Top-N_{seed}.json"

with open(metrics_path, 'rb') as f:
    metrics = load(f)

# print(metrics.keys())

print(metrics["Chosen"][0][0][:20], metrics["Chosen"][0][1][:20])

candels = "candels" in dataset

if candels:
    ds__ = "candels"
else:
    ds__ = dataset

embedding = np.load(f"{ds__}_{emb}_.npy")
data = np.load(f"{ds__}_X.npy")

labels = np.load(f"{ds__}_Y.npy")

idxs_chosen = np.zeros(len(data), dtype=np.uint8)
idxs_orig = np.zeros(len(data), dtype=np.uint8)

colors_edge = {
    "0": None,  # unlabeled
    "1": "rgb(0, 26, 255)",  # default_chosen
}

colors_fill = {
    "0": "rgb(122, 122, 122)",  # unlabeled
    "1": "rgb(255, 1, 1)",  # prev_chosen
    "2": "rgb(255, 1, 1)",  # updated_chosen
}

source = ColumnDataSource(data=dict(x=embedding[:, 0], y=embedding[:, 1],
                                    fill_color=[colors_fill[f"{x}"] if x != 1 else 0 for x in idxs_chosen],
                                    line_color=[colors_edge[f"{x}"] if x == 1 else 0 for x in idxs_orig]))

embed = pn.Row(sizing_mode="stretch_both")

image = pn.layout.Row()
plot = figure(width=750, height=750)
plot.circle('x', 'y', fill_color="fill_color", line_color="line_color", source=source)

plots = pn.pane.Bokeh(plot)

embed.objects = [pn.layout.Row(plots, pn.Column(image))]

s1 = ColumnDataSource(data=dict(idx=[0]))

callback = CustomJS(args=dict(s1=s1),
                    code="""


                        var current_idx = s1.data.idx[0];
                        if (cb_data['index'].indices.length > 0){
                            if(cb_data['index'].indices[0] != current_idx){

                            var dict = {
                                idx: cb_data['index'].indices,
                                };


                            s1.data = dict;
                            }
                        }



                        """)

hover_tool_plot = HoverTool(callback=callback)


def get_values(attr, old, new):
    img = data[s1.data["idx"][0]]

    video = pn.pane.Video('/mnt/6B2B9F977D3D20FC/Uni-Stuff/PhD/Deep-Active-Learning/talk.mp4', width=640, loop=True,
                          autoplay=True)

    image.objects = [pn.layout.Column(video)]


s1.on_change("data", get_values)

plot.add_tools(hover_tool_plot)

template = pn.template.FastGridTemplate(site="Panel", title="App", prevent_collision=True)

template.main[0:12, 1:12] = embed
template.servable()
