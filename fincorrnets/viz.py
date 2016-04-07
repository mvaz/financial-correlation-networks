from bokeh.plotting import *
from bokeh.objects import HoverTool, ColumnDataSource
from bokeh.sampledata.les_mis import data

from collections import OrderedDict

import scipy.cluster.hierarchy as sch
import seaborn as sns

import numpy as np

from IPython.html import widgets

import networkx as nx


def prepare_datasources(G, **kwargs):
    layout = nx.layout.spring_layout(G, **kwargs)
    import community
    partition = community.best_partition(G)
    df_ = pd.DataFrame(layout).T
    df_.columns = ['x', 'y']
    df_['community'] = pd.DataFrame.from_dict(
        {'nodes': list(partition.keys()), 'community': list(partition.values())}).set_index('nodes')
    n_communities = df_.community.max() + 1
    df_['color'] = df_.community.apply(hex_palette(n_communities))

    ctr = compute_centrality_metrics(G)
    df_['size'] = 8 * ctr.XpY
    df_['line_color'] = 'navy'
    chosen = ctr.XpY >= ctr.XpY.quantile((ctr.XpY.size - 30.) / ctr.XpY.size)
    df_['color'][chosen] = 'red'
    ds_nodes = ColumnDataSource(df_)

    df_edges = pd.DataFrame((df_.x[e[0]], df_.y[e[0]], df_.x[e[1]], df_.y[e[1]]) for e in G.edges_iter())
    df_edges.columns = ['x0', 'y0', 'x1', 'y1']
    ds_edges = ColumnDataSource(df_edges)

    return ds_nodes, ds_edges


def plot_graph(p, ds_edges, ds_nodes):
    segments = p.segment('x0', 'y0', 'x1', 'y1', color='grey', line_width=1, alpha=0.3, source=ds_edges)
    circles = p.scatter('x', 'y', marker='circle', size='size', line_color="navy", fill_color="color", alpha=0.8,
                        source=ds_nodes)
    return circles, segments


class MatrixPlotter(object):
    """docstring for MatrixPlotter"""

    def __init__(self, source):
        super(MatrixPlotter, self).__init__()
        self.number_colors = 21
        self.source = source
        self.column_source = None
        self.plot = None
        self.palette = None
        self._init_palette()

    def corrplot(self, entities):
        figure()
        rect('xname', 'yname', 0.9, 0.9, source=self.column_source,
             x_range=entities, y_range=list(reversed(entities)),
             color='colors', line_color=None,
             tools="resize,hover", title="Correlation matrix",
             plot_width=500, plot_height=500)
        grid().grid_line_color = None
        axis().axis_line_color = None
        axis().major_tick_line_color = None
        axis().major_label_text_font_size = "7pt"
        axis().major_label_standoff = 0

        xaxis().location = "top"
        xaxis().major_label_orientation = np.pi / 3
        self.plot = curplot()

        # hover = [t for t in curplot().tools if isinstance(t, HoverTool)][0]
        hover = [t for t in self.plot.tools if isinstance(t, HoverTool)][0]
        hover.tooltips = OrderedDict([
            ('names', '@yname, @xname'),
            ('count', '@values')
        ])
        return self

    @staticmethod
    def reorder_dendogram(df):
        Y = sch.linkage(df.values, method='centroid')
        Z = sch.dendrogram(Y, orientation='right', no_plot=True)
        index = Z['leaves']
        return index

    def _init_palette(self):
        basis = sns.blend_palette(["seagreen", "ghostwhite", "#4168B7"], self.number_colors)
        self.palette = ["rgb(%d, %d, %d)" % (r, g, b) for r, g, b, a in np.round(basis * 255)]

    def _color(self, value):
        i = np.round((value + 1.) * (self.number_colors - 1) * 0.5)
        return self.palette[int(i)]

    def to_data_source(self, df):
        index = self.reorder_dendogram(df)
        # col = lambda v: self.color(v)
        print self._color(0.2)
        _names = df.columns.tolist()

        names = [_names[i] for i in index]
        xnames = []
        ynames = []
        values = []
        colors = []
        for n in names:
            xnames.extend([n] * len(names))
            ynames.extend(names)
            v = df.loc[n, names].tolist()
            values.extend(values)
            colors.extend([self._color(x) for x in v])
        # alphas = np.abs(df.values).flatten()
        self.column_source = ColumnDataSource(
            data=dict(
                xname=xnames,
                yname=ynames,
                colors=colors,
                values=values,
            )
        )
        return self, names

    def as_widget(self):
        bokeh_widget = widgets.HTMLWidget()
        bokeh_widget.value = notebook_div(self.plot)
        return bokeh_widget
