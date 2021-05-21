import math
from typing import Callable
import numpy as np

import plotly as py
import plotly.graph_objs as go
pyplt = py.offline.plot


class gaussian_pdf:
    """高斯分布概率密度函数
    """
    def __init__(self, mu: float, sigma: float) -> None:
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x: float) -> float:
        return math.exp((-((x-self.mu)/self.sigma)**2)/2)/(self.sigma*math.sqrt(2*math.pi))


def func_cdf(pdf_func: Callable[[float], float], start: float, x: float, step=0.01) -> float:
    """累积分布函数
    对概率密度函数利用梯形面积法进行近似积分

    Attributes:
        pdf_func: 概率密度函数
        start: 积分起点
        x: 积分终点
        step: 步长（近似梯形的高，值越小结果越准确，耗时越长）

    Reutrns:
        x处的累积分布函数值
    """
    res = 0.0
    # 梯形下底X坐标(左侧边界)
    bottom_x = start
    # 梯形上底X坐标(右侧边界)
    top_x = bottom_x + step
    while top_x < x:
        # 下底高度
        bottom_y = pdf_func(bottom_x)
        # 上底高度
        top_y = pdf_func(top_x)
        # 累积梯形面积
        res += (bottom_y + top_y) * step / 2
        # 向后移动
        bottom_x += step
        top_x += step
    return res


def func_cdfinv(pdf_func: Callable[[float], float], start: float,
                left: float, right: float, y: float, step=0.0001, precision=0.0001) -> float:
    """累积分布函数逆函数
    借助累积分布函数，二分搜索确定逆函数的值（累积分布函数应为严格增函数）

    Attributes:
        pdf_func: 概率密度函数
        start: 积分起点
        left: 二分搜索左侧边界
        right: 二分搜索右侧边界
        y: 累积分布函数值
        step: 步长（近似梯形的高，值越小结果越准确，耗时越长）
        precision: 结果精度

    Reutrns:
        y处的累积分布函数反函数值
    """
    left_x = left
    right_x = right
    left_y = func_cdf(pdf_func, start, left_x, step)
    right_y = func_cdf(pdf_func, start, right_x, step)
    if abs(left_y - right_y) < precision:
        return (left_x + right_y) / 2

    mid_x = (left_x + right_x) / 2
    mid_y = func_cdf(pdf_func, start, mid_x, step)

    if (y <= mid_y):
        return func_cdfinv(pdf_func, start, left_x, mid_x, y, step, precision)
    else:
        return func_cdfinv(pdf_func, start, mid_x+step, right_x, y, step, precision)


if __name__ == '__main__':
    mu = 0
    sigma = 1
    start = mu-5*sigma
    end = mu+5*sigma
    pdf = gaussian_pdf(mu, sigma)
    x = np.arange(start, end, 0.01)
    pdf_y = [pdf(t) for t in x]
    cdf_y = [func_cdf(pdf, start, t) for t in x]

    inv_x = np.arange(0, 1, 0.01)
    cdfinv_y = [func_cdfinv(pdf, start, start, end, t, 0.01, 0.01) for t in inv_x]

    trace0 = go.Scatter(x=x, y=pdf_y, name='GaussianPDF')
    trace1 = go.Scatter(x=x, y=cdf_y, name='GaussianCDF')
    trace2 = go.Scatter(x=inv_x, y=cdfinv_y, name='GaussianCDFINV')
    data = [trace0, trace1, trace2]
    layout = go.Layout(
        title="Gaussian",
        xaxis=dict(title='x'),
        yaxis=dict(title='y'),        
    )
    fig = go.Figure(data, layout)
    pyplt(fig, filename='gaussian.html')
