import scipy.stats as st
import numpy as np
import plotly as py
import plotly.graph_objs as go
pyplt = py.offline.plot


if __name__ == '__main__':
    mu = 0
    sigma = 1
    start = mu-5*sigma
    end = mu+5*sigma
    x = np.arange(start, end, 0.01)
    pdf_y = st.norm.pdf(x)
    cdf_y = st.norm.cdf(x)

    inv_x = np.arange(0, 1, 0.01)
    cdfinv_y = st.norm.ppf(inv_x)

    rand_x = np.random.uniform(0, 1, 100000)
    rand_y = np.round(st.norm.ppf(rand_x), 1)

    unique, count = np.unique(rand_y, return_counts=True)

    trace0 = go.Scatter(x=x, y=pdf_y, name='GaussianPDF')
    trace1 = go.Scatter(x=x, y=cdf_y, name='GaussianCDF')
    trace2 = go.Scatter(x=inv_x, y=cdfinv_y, name='GaussianCDFINV')

    trace3 = go.Scatter(x=unique, y=count, name='GaussianCDFINV')

    data = [trace0, trace1, trace2]
    layout = go.Layout(
        title="Gaussian",
        xaxis=dict(title='x'),
        yaxis=dict(title='y'),
    )
    fig = go.Figure(data, layout)
    pyplt(fig, filename='gaussian.html')

    data2 = [trace3]
    layout2 = go.Layout(
        title="Gaussian",
        xaxis=dict(title='x'),
        yaxis=dict(title='y'),
    )
    fig2 = go.Figure(data2, layout2)
    pyplt(fig2, filename='gaussian2.html')
