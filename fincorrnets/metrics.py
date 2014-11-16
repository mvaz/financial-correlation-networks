import networkx as nx
import pandas as pd


def calc_metric(metric, metric_name, g, **kwargs):
    return pd.DataFrame.from_dict( metric(g, **kwargs), orient='index').rename(columns={0: metric_name})

def compute_rank(col, order='ascending'):
.    return col.rank(ascending=(order == 'ascending'))


def compute_rank(col, order='ascending'):
    u, v = np.unique(col if order == 'ascending' else -col, return_inverse=True)
    return (np.cumsum(np.bincount(v, minlength=u.size)) - 1)[v]

def betweenness_centrality(g, weighted=True):
    _name = 'betweenness' + ('_weighted' if weighted else '')
    _d = calc_metric(nx.algorithms.centrality.betweenness_centrality, _name, g, weight='weight' if weighted else None)
    _d[_name] = compute_rank(_d[_name], order='descending')
    return _d

def degree_centrality(g, weighted=True):
    # TODO weight not yet taken into account, make own function
    _name = 'degree' + ('_weighted' if weighted else '')
    _d = calc_metric(nx.algorithms.centrality.degree_centrality, _name, g ) #, weight='weight' if weighted else None)
    _d[_name] = compute_rank(_d[_name], order='descending')
    return _d


def eigenvector_centrality(g, weighted=True):
    _name = 'eigenvector' + ('_weighted' if weighted else '')
    _d = calc_metric(nx.algorithms.centrality.eigenvector_centrality_numpy, _name, g, weight='weight' if weighted else None)
    _d[_name] = compute_rank(_d[_name], order='descending')
    return _d

def eccentricity(g, weighted=True):
    ### weight not yet taken into accoutn
    _name = 'eccentricity' + ('_weighted' if weighted else '')
    _d = calc_metric(nx.algorithms.distance_measures.eccentricity, _name, g) #, weight='weight' if weighted else None)
    _d[_name] = compute_rank(_d[_name], order='ascending')
    return _d

def closeness(g, weighted=True):
    def calc2(metric, metric_name, g, distance='weight', **kwargs):
        return pd.DataFrame.from_dict( metric(g, distance=distance), orient='index').rename(columns={0: metric_name})
    _name = 'closeness' + ('_weighted' if weighted else '')
    _d = calc_metric(nx.algorithms.centrality.closeness.closeness_centrality, _name, g, distance='weight' if weighted else None)
    _d[_name] = compute_rank(_d[_name], order='ascending')
    return _d



def compute_centrality_metrics(g):
    """  Computes the centrality metrics according to ... """
    c = betweenness_centrality(g, weighted=True)
    # c['betweenness']
    c['betweenness'] = betweenness_centrality(g, weighted=False)
    c['degree'] = degree_centrality(g, weighted=False)
    c['degree_weighted'] = degree_centrality(g, weighted=True)
    c['X'] = (c.betweenness + c.betweenness_weighted + c.degree + c.degree_weighted  - 4) / 4 / len(c.index)

    c['eigenvector_weighted'] = eigenvector_centrality(g)
    c['eigenvector'] = eigenvector_centrality(g, weighted=False)
    c['closeness_weighted'] = closeness(g)
    c['closeness'] = closeness(g, weighted=False)
    c['eccentricity_weighted'] = eccentricity(g)
    c['eccentricity'] = eccentricity(g, weighted=False)
    c['Y'] = (c.eccentricity_weighted + c.eigenvector + c.closeness + c.closeness_weighted + c.eccentricity + c.eccentricity_weighted - 6) / 6 / len(c.index)
    c['XmY'] = c.Y - c.X
    c['XpY'] = c.Y + c.X
    return c