import random
from scipy.optimize import minimize
import numpy as np
import mpmath

class wienerModel():
    def __init__(self, _x, _trend, _z0):
        self.x = _x
        self.trend = _trend
        self.z = _z0
        self.data = [[]]

    def setParam(self, _x):
        self.x = _x

    def f_ro(self, t, c=0):
        return self.trend.func(self.x, t, c)

    def f_trend(self,t, c=0):
        return x[1]*self.f_ro(self.x, t, c)

    def increment(self, delta_ro):
        tetha1 = self.x[1]*delta_ro
        tetha2 = self.x[0]*np.sqrt(delta_ro)
        return random.normalvariate(tetha1,tetha2)

    def estimateParameters(self, x0, _data):
        self.data = _data
        res = minimize(self.LNf, x0, method='nelder-mead')
        res = res.x
        return res

    def LNf(self, x):
        time = self.data[0]
        delta = self.data[1]
        cov = self.data[3]
        n = len(time)
        f = 0
        self.setParam(x)
        for i in range(n):
            f += (len(time[i]) - 1) * (0.5 * np.log(2 * np.pi) + np.log(x[0]))
            s1, s2 = 0, 0
            for j in range(len(time[i]) - 1):
                delta_ro = self.f_ro(time[i][j + 1], cov[i]) - self.f_ro(time[i][j], cov[i])
                tetha1 = x[1] * delta_ro
                tetha2 = x[0] * np.sqrt(2 * delta_ro)
                if (tetha1 <= 0 or tetha2 <= 0):
                    return 1e+10
                s1 += np.power((delta[i][j] - tetha1) / tetha2, 2)
                s2 += np.log(delta_ro) / 2
            f += s1 + s2
        return f

    def estimateConditionalParameters(self, x0, _data):
        self.data = _data
        res = minimize(self.LNf_cr, x0, method='nelder-mead')
        res = res.x
        return res

    def LNf_cr(self, x):
        z = self.z
        time = self.data[0]
        delta = self.data[1]
        covariates = self.data[3]

        n = len(time)
        um2 = 0
        f = 0
        self.setParam(x)

        for i in range(n):

            if (x[0] <= 0 or x[1] <= 0):
                return 1e+10

            f += 0.5 * np.log(2 * np.pi) + np.log(x[0])
            ro = self.f_ro(time[i][-1], covariates[i])
            tetha1 = x[1] * ro
            tetha2 = np.power(x[0], 2) * 2 * ro
            f += np.power((sum(delta[i]) - tetha1), 2) / tetha2
            f += np.log(ro) / 2

            sum2 = float(mpmath.gammainc(0.5, np.power((z - tetha1) / tetha2, 2)))
            # print(sum2)
            um2 += np.log(0.5 + 0.5 * sum2 / (np.sqrt(np.pi)))

        return f + um2

