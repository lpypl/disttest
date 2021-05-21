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


class cdf_type:
    """累积分布函数类
    对概率密度函数利用梯形面积法进行近似积分

    Reutrns:
        x处的累积分布函数值
    """

    def __init__(self, pdf_func: Callable[[float], float], start: float, end: float, step=0.01) -> None:
        """
        Attributes:
            pdf_func: 概率密度函数
            start: 积分起点
            x: 积分终点
            step: 步长（近似梯形的高，值越小结果越准确，耗时越长）
        """
        self.pdf_func = pdf_func
        self.start = start
        self.end = end
        self.step = step

    def __call__(self, x: float) -> float:
        """
        Attributes:
            x: 积分终点

        Reutrns:
            x处的累积分布函数值
        """
        res = 0.0
        # 梯形下底X坐标(左侧边界)
        bottom_x = self.start
        # 梯形上底X坐标(右侧边界)
        top_x = bottom_x + self.step
        while top_x < x:
            # 下底高度
            bottom_y = self.pdf_func(bottom_x)
            # 上底高度
            top_y = self.pdf_func(top_x)
            # 累积梯形面积
            res += (bottom_y + top_y) * self.step / 2
            # 向后移动
            bottom_x += self.step
            top_x += self.step
        return res


class cdfinv_type:
    """累积分布函数逆函数
    借助累积分布函数，二分搜索确定逆函数的值（累积分布函数应为严格增函数）

    Reutrns:
        y处的累积分布函数反函数值
    """

    def __init__(self, cdf_func: cdf_type, precision=0.01) -> None:
        """
        Attributes:
            pdf_func: 概率密度函数
            start: 积分起点
            left: 二分搜索左侧边界
            right: 二分搜索右侧边界
            y: 累积分布函数值
            step: 步长（近似梯形的高，值越小结果越准确，耗时越长）
            precision: 结果精度
        """
        self.cdf_func = cdf_func
        self.precision = precision

    def __call__(self, y: float) -> float:
        """
        Attributes:
            y: 累积分布函数值

        Reutrns:
            y处的累积分布函数反函数值
        """
        return self.inv(self.cdf_func.start, self.cdf_func.end, y)

    def inv(self, left: float, right: float, y: float):
        """累积分布函数逆函数
        借助累积分布函数，二分搜索确定逆函数的值（累积分布函数应为严格增函数）

        Attributes:
            left: 二分搜索左侧边界
            right: 二分搜索右侧边界
            y: 累积分布函数值

        Reutrns:
            y处的累积分布函数反函数值
        """
        left_x = left
        right_x = right
        left_y = self.cdf_func(left_x)
        right_y = self.cdf_func(right_x)
        
        if abs(left_y - right_y) < self.precision:
            return (left_x + right_x) / 2

        mid_x = (left_x + right_x) / 2
        mid_y = self.cdf_func(mid_x)

        if (y <= mid_y):
            return self.inv(left_x, mid_x, y)
        else:
            return self.inv(mid_x, right_x, y)



# if __name__ == '__main__':
#     mu = 0
#     sigma = 1
#     start = mu-5*sigma
#     end = mu+5*sigma
#     pdf = gaussian_pdf(mu, sigma)
#     cdf = cdf_type(pdf, start, end)
#     cdfinv = cdfinv_type(cdf) 
#     x = np.arange(start, end, 0.01)
#     pdf_y = [pdf(t) for t in x]
#     cdf_y = [cdf(t) for t in x]

#     inv_x = np.arange(0, 1, 0.01)
#     cdfinv_y = [cdfinv(t)
#                 for t in inv_x]

#     trace0 = go.Scatter(x=x, y=pdf_y, name='GaussianPDF')
#     trace1 = go.Scatter(x=x, y=cdf_y, name='GaussianCDF')
#     trace2 = go.Scatter(x=inv_x, y=cdfinv_y, name='GaussianCDFINV')
#     data = [trace0, trace1, trace2]
#     layout = go.Layout(
#         title="Gaussian",
#         xaxis=dict(title='x'),
#         yaxis=dict(title='y'),
#     )
#     fig = go.Figure(data, layout)
#     pyplt(fig, filename='gaussian.html')


# if __name__ == '__main__':
#     mu = 0
#     sigma = 1
#     start = mu-10*sigma
#     end = mu+10*sigma
#     pdf = gaussian_pdf(mu, sigma)
#     cdf = cdf_type(pdf, start, end, step=0.001)
#     cdfinv = cdfinv_type(cdf, precision=0.001) 

#     tablesize = 167
#     randx = np.random.uniform()
#     randy = cdfinv(randx)

#     print(f'randx is {randx}')
#     print(f'randy is {randy}')
#     print(f'invrandx is {cdf(randy)}')
#     print(f'invinvrandx is {cdfinv(cdf(randy))}')

if __name__ == '__main__':
    tablesize = 167
    mu = 167/2
    sigma = mu/3
    start = mu-10*sigma
    end = mu+10*sigma
    pdf = gaussian_pdf(mu, sigma)
    cdf = cdf_type(pdf, start, end, step=0.01)
    cdfinv = cdfinv_type(cdf, precision=0.001)

    tablesize = 167
    for randx in range(1, tablesize):
        randy = cdfinv(np.round(1/randx, 3))
        invrandx = np.round(1/cdf(randy), 0)
        print(f'{randx}\t{randy}\t{invrandx}')
