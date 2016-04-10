__author__ = 'Miguel Vaz'

import numpy as np
import pandas as  pd

import networkx as nx
from bokeh.objects import HoverTool, ColumnDataSource


def edges_datasource(g, layout=None, line_width=None, line_color=None):
    """
    Creates a bokeh ColumnDataSource with the computed parameters for drawing.
    Namely, it sets the values 'x0', 'x1', 'y0', 'y1', 'line_width', 'line_color' for each of the edges of the graph.

    :param g: a networkx graph to draw
    :param layout: a layout dictionary, as computed by he networkx.layout functions
    :param line_width:
    :param line_color:
    :return: a ColumnDataSource and a corresponding DataFrame, where each row is a
    """
    if layout is None:
        layout = nx.layout.spring_layout(g)

    df_layout = pd.DataFrame(layout, index=['x', 'y']).T

    if line_width is None:
        f_weight = lambda x: x
    elif hasattr(line_width, '__call__'):
        f_weight = line_width
    else:
        f_weight = lambda x: line_width

    if line_color is None:
        f_color = lambda x: x
    elif hasattr(line_color, '__call__'):
        f_color = line_color
    else:
        f_color = lambda x: line_color

    df_edges = pd.DataFrame(((df_layout.x[e[0]],
                              df_layout.y[e[0]],
                              df_layout.x[e[1]],
                              df_layout.y[e[1]],
                              f_weight(e[2].get('weight', 1)),
                              f_color(e[2].get('weight', 1))
                              ) for e in g.edges_iter(data=True, default={'weight': 1})),
                            columns=['x0', 'y0', 'x1', 'y1', 'line_width', 'line_color'])

    return ColumnDataSource(df_edges), df_edges


def nodes_datasource(g, layout=None):
    """

    :param g:
    :param layout:
    :return:
    """
    if layout is None:
        layout = nx.layout.spring_layout(g)

    df_layout = pd.DataFrame(layout, index=['x', 'y']).T
    df_layout['size'] = 10

    return ColumnDataSource(df_layout), df_layout


def plot_edges(fig, ds=None, line_alpha=0.3):
    """

    :param fig:
    :param ds:
    :param line_alpha:
    :return:
    """
    segments = fig.segment('x0', 'y0', 'x1', 'y1', color='grey', line_width='line_width', alpha=line_alpha, source=ds)
    return segments


def plot_nodes(fig, ds=ds_nodes):
    """

    :param fig:
    :param ds:
    :return:
    """
    circles = p.scatter('x', 'y', marker='circle', size='size', line_color="navy", fill_color="color", alpha=0.8,
                        source=ds)
    return circles


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
