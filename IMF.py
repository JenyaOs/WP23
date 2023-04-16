import degradationData as data
import wienerModel as model
import mpmath
import numpy as np

class Wiener_IMF():
    
    def __init__(self, _model, _n, _cov, _weight, _time):
        self.model = _model
        self.covariates = self.convertToArray(_cov,_weight,_n)
        self.weight = _weight
        self.time = _time
        
    def getIMF(self):
        if(self.model.trend.get_type()=='linearTrend'):
            return [[self.I11(),self.I12()],[self.I12(),self.I22()]]
        if (self.model.trend.get_type() == 'linearTrendWithCovariate'):
            return [[self.I11(),self.I12(),self.I14()],
                    [self.I12(),self.I22(),self.I24()],
                    [self.I14(),self.I24(),self.I44()]]
        if (self.model.trend.get_type() == 'powerTrend'):
            return [[self.I11(),self.I12(),self.I13()],
                    [self.I12(),self.I22(),self.I23()],
                    [self.I13(),self.I23(),self.I33()]]
        if (self.model.trend.get_type() == 'powerTrendWithCovariate'):
            return [[self.I11(),self.I12(),self.I13(),self.I14()],
                    [self.I12(),self.I22(),self.I23(),self.I24()],
                    [self.I13(),self.I23(),self.I33(),self.I34()],
                    [self.I14(),self.I24(),self.I34(),self.I44()]]

    def convertToArray(self, cov, weight, n):
        array = []
        for i in range(len(cov)-1):
            for j in range(int(weight[i]*n)):
                array.append(cov[i])
        for k in range(n - len(array)):
            array.append(cov[-1])
        return array
    
    def I11(self):
        time = self.time
        n = len(time)
        x = self.model.x
        return 2 * sum([len(time[i]) - 1 for i in range(n)]) / np.power(x[0], 2)

    def I12(self):
        return 0
    
    def I13(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates

        s = 0
        for i in range(n):
            for j in range(len(time[i]) - 1):
                delta_ro = self.model.f_ro(time[i][j + 1], cov[i]) - self.model.f_ro(time[i][j], cov[i])
                der_ro = self.model.trend.d_func_gamma(time[i][j + 1], x, cov[i]) - self.model.trend.d_func_gamma(time[i][j], x,
                                                                                                      cov[i])
                s += der_ro / delta_ro
        return s / x[0]

    def I14(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates

        s = 0
        for i in range(n):
            for j in range(len(time[i]) - 1):
                delta_ro = self.model.f_ro(time[i][j + 1],  cov[i]) - self.model.f_ro(time[i][j], cov[i])
                der_ro = self.model.trend.d_func_beta(time[i][j + 1], x, cov[i]) - self.model.trend.d_func_beta(time[i][j], x,  cov[i])
                s += der_ro / delta_ro

        return s / x[0]
    
    def I22(self):
        time = self.time
        n = len(time)
        x = self.model.x
        s = 0
        cov = self.covariates

        for i in range(n):
            for j in range(len(time[i]) - 1):
                delta_ro = self.model.f_ro(time[i][j + 1], cov[i]) - self.model.f_ro(time[i][j], cov[i])
                s += delta_ro
        return s / np.power(x[0], 2)
    
    def I23(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates

        s = 0
        for i in range(n):
            for j in range(len(time[i]) - 1):
                der_ro = self.model.trend.d_func_gamma(time[i][j + 1],x, cov[i]) - self.model.trend.d_func_gamma(time[i][j],x,
                                                                                                      cov[i])
                s += der_ro
        return s * np.power(x[1] / x[0], 2)

    def I24(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates

        s = 0
        for i in range(n):
            for j in range(len(time[i]) - 1):
                der_ro = self.model.trend.d_func_beta(time[i][j + 1], x, cov[i]) \
                         - self.model.trend.d_func_beta(time[i][j], x, cov[i])
                s += der_ro
        return s * np.power(x[1] / x[0], 2)
    
    def I33(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates
        
        s = 0
        for i in range(n):
            for j in range(len(time[i]) - 1):
                delta_ro = self.model.f_ro(time[i][j + 1], cov[i]) - self.model.f_ro(time[i][j],  cov[i])
                der_ro = self.model.trend.d_func_gamma(time[i][j + 1], x, cov[i]) - self.model.trend.d_func_gamma(time[i][j], x,
                                                                                                      cov[i])
                s += np.power(der_ro, 2) * (.5 / np.power(delta_ro, 2) + np.power(x[1] / x[0], 2) / delta_ro)
        return s 
    
    def I34(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates
        
        s = 0
        for i in range(n):
            for j in range(len(time[i]) - 1):
                delta_ro = self.model.f_ro(time[i][j + 1],  cov[i]) - self.model.f_ro(time[i][j], cov[i])
                der_ro_g = self.model.trend.d_func_gamma(time[i][j + 1], x, cov[i]) - self.model.trend.d_func_gamma(time[i][j], x,
                                                                                                        cov[i])
                der_ro_b = self.model.trend.d_func_beta(time[i][j + 1], x, cov[i]) - self.model.trend.d_func_beta(time[i][j], x,
                                                                                                      cov[i])
                s += der_ro_g * der_ro_b * (.5 / np.power(delta_ro, 2) + np.power(x[1] / x[0], 2) / delta_ro)
        return s
    
    def I44(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates
        
        s = 0
        for i in range(n):
            for j in range(len(time[i]) - 1):
                delta_ro = self.model.f_ro(time[i][j + 1], cov[i]) - self.model.f_ro(time[i][j],  cov[i])
                der_ro = self.model.trend.d_func_beta(time[i][j + 1], x, cov[i]) - self.model.trend.d_func_beta(time[i][j], x,
                                                                                                    cov[i])
                s += np.power(der_ro, 2) * (.5 / np.power(delta_ro, 2) + np.power(x[1] / x[0], 2) / delta_ro)
        return s

class Wiener_ConditionIMF():

    def __init__(self, _model, _n, _cov, _weight, _time, _z0):
        self.model = _model
        self.covariates = self.convertToArray(_cov, _weight, _n)
        self.weight = _weight
        self.time = _time
        self.z0 = _z0

    def getIMF(self):
        IMF1 = np.array(self.getIMF_part1())
        IMF2 =  np.array(self.getIMF_part2())
        #print(IMF1, '\n', IMF2)
        return IMF1 + IMF2


    def getIMF_part1(self):
        if (self.model.trend.get_type() == 'linearTrend'):
            return [[self.I11(), self.I12()], [self.I12(), self.I22()]]
        if (self.model.trend.get_type() == 'linearTrendWithCovariate'):
            return [[self.I11(), self.I12(), self.I14()],
                    [self.I12(), self.I22(), self.I24()],
                    [self.I14(), self.I24(), self.I44()]]
        if (self.model.trend.get_type() == 'powerTrend'):
            return [[self.I11(), self.I12(), self.I13()],
                    [self.I12(), self.I22(), self.I23()],
                    [self.I13(), self.I23(), self.I33()]]
        if (self.model.trend.get_type() == 'powerTrendWithCovariate'):
            return [[self.I11(), self.I12(), self.I13(), self.I14()],
                    [self.I12(), self.I22(), self.I23(), self.I24()],
                    [self.I13(), self.I23(), self.I33(), self.I34()],
                    [self.I14(), self.I24(), self.I34(), self.I44()]]

    def getIMF_part2(self):
        if (self.model.trend.get_type() == 'linearTrend'):
            return [[self.I11_F(), self.I12_F()], [self.I12_F(), self.I22_F()]]
        if (self.model.trend.get_type() == 'linearTrendWithCovariate'):
            return [[self.I11_F(), self.I12_F(), self.I14_F()],
                    [self.I12_F(), self.I22_F(), self.I24_F()],
                    [self.I14_F(), self.I24_F(), self.I44_F()]]
        if (self.model.trend.get_type() == 'powerTrend'):
            return [[self.I11_F(), self.I12_F(), self.I13_F()],
                    [self.I12_F(), self.I22_F(), self.I23_F()],
                    [self.I13_F(), self.I23_F(), self.I33_F()]]
        if (self.model.trend.get_type() == 'powerTrendWithCovariate'):
            return [[self.I11_F(), self.I12_F(), self.I13_F(), self.I14_F()],
                    [self.I12_F(), self.I22_F(), self.I23_F(), self.I24_F()],
                    [self.I13_F(), self.I23_F(), self.I33_F(), self.I34_F()],
                    [self.I14_F(), self.I24_F(), self.I34_F(), self.I44_F()]]

    def convertToArray(self, cov, weight, n):
        array = []
        for i in range(len(cov) - 1):
            for j in range(int(weight[i] * n)):
                array.append(cov[i])
        for k in range(n - len(array)):
            array.append(cov[-1])
        return array

    def I11(self):
        time = self.time
        n = len(time)
        x = self.model.x
        return 2 * n / np.power(x[0], 2)

    def I12(self):
        return 0

    def I13(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates

        s = 0
        for i in range(n):
                delta_ro = self.model.f_ro(time[i][-1], cov[i])
                der_ro = self.model.trend.d_func_gamma(time[i][-1], x, cov[i])
                s += der_ro / delta_ro
        return s / x[0]

    def I14(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates

        s = 0
        for i in range(n):
                delta_ro = self.model.f_ro(time[i][-1], cov[i])
                der_ro = self.model.trend.d_func_beta(time[i][-1], x, cov[i])
                s += der_ro / delta_ro

        return s / x[0]

    def I22(self):
        time = self.time
        n = len(time)
        x = self.model.x
        s = 0
        cov = self.covariates

        for i in range(n):
                delta_ro = self.model.f_ro(time[i][-1], cov[i])
                s += delta_ro
        return s / np.power(x[0], 2)

    def I23(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates

        s = 0
        for i in range(n):
                der_ro = self.model.trend.d_func_gamma(time[i][-1], x, cov[i])
                s += der_ro
        return s * np.power(x[1] / x[0], 2)

    def I24(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates

        s = 0
        for i in range(n):
            der_ro = self.model.trend.d_func_beta(time[i][- 1], x, cov[i])
            s += der_ro
        return s * np.power(x[1] / x[0], 2)

    def I33(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates

        s = 0
        for i in range(n):
                delta_ro = self.model.f_ro(time[i][-1], cov[i])
                der_ro = self.model.trend.d_func_gamma(time[i][-1], x, cov[i])
                s += np.power(der_ro, 2) * (.5 / np.power(delta_ro, 2) + np.power(x[1] / x[0], 2) / delta_ro)
        return s

    def I34(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates

        s = 0
        for i in range(n):
                delta_ro = self.model.f_ro(time[i][-1],  cov[i])
                der_ro_g = self.model.trend.d_func_gamma(time[i][-1], x, cov[i])
                der_ro_b = self.model.trend.d_func_beta(time[i][-1], x, cov[i])
                s += der_ro_g * der_ro_b * (.5 / np.power(delta_ro, 2) + np.power(x[1] / x[0], 2) / delta_ro)
        return s

    def I44(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates

        s = 0
        for i in range(n):
                delta_ro = self.model.f_ro(time[i][- 1], cov[i])
                der_ro = self.model.trend.d_func_beta(time[i][-1], x, cov[i])
                s += np.power(der_ro, 2) * (.5 / np.power(delta_ro, 2) + np.power(x[1] / x[0], 2) / delta_ro)
        return s


    def F(self, i):
        time = self.time[i]
        x = self.model.x
        cov = self.covariates
        z0 = self.z0

        s = .5 + (0.5 / np.sqrt(np.pi)) * \
            mpmath.gammainc(.5, np.power(z0 - x[1] * self.model.f_ro(time[-1], cov[i]), 2) /
                            (2 * pow(x[0], 2) * self.model.f_ro(time[-1], cov[i])))
        return s

    def C(self,i):
        time = self.time
        x = self.model.x
        cov = self.covariates
        z0 = self.z0

        return (z0 - x[1] * self.model.f_ro(time[i][- 1], cov[i])) / (x[0] * np.sqrt(2 * self.model.f_ro(time[i][- 1], cov[i])))

    def d_C(self, i, k):
        time = self.time
        x = self.model.x
        cov = self.covariates
        z0 = self.z0

        if (k == 0):
            return -(z0 - x[1] * self.model.f_ro(time[i][-1],  cov[i])) / (
                    np.power(x[0], 2) * np.sqrt(2 * self.model.f_ro(time[i][-1],  cov[i])))
        if (k == 1):
            return - np.sqrt(self.model.f_ro(time[i][-1],  cov[i])) / (np.sqrt(2) * x[0])

        if (k == 2):
            return - self.model.trend.d_func_gamma(time[i][-1], x, cov[i]) * (
                    x[1] + z0 / self.model.f_ro(time[i][-1],  cov[i])) / (
                    x[0] * 2 * np.sqrt(2 * self.model.f_ro(time[i][-1],  cov[i])))
        if (k == 3):
            return - self.model.trend.d_func_beta(time[i][-1], x, cov[i]) * (
                    x[1] + z0 / self.model.f_ro(time[i][-1],  cov[i])) / (
                    x[0] * 2 * np.sqrt(2 * self.model.f_ro(time[i][-1],  cov[i])))

    def d2_C(self, i, k, m):

        time = self.time[i]
        x = self.model.x
        c  = self.covariates[i]
        z0 = self.z0 
        
        if (k == 0 and m == 0):
            return 2 * (z0 - x[1] * self.model.f_ro(time[-1], c)) / (
                        np.power(x[0], 3) * np.sqrt(2 * self.model.f_ro(time[-1], c)))

        if (k == 0 and m == 1) or (k == 1 and m == 0):
            return np.sqrt(self.model.f_ro(time[-1], c) / 2) / np.power(x[0], 2)

        if (k == 0 and m == 2) or (k == 2 and m == 0):
            return self.model.trend.d_func_gamma(time[-1], x, c) * (x[1] + z0 / self.model.f_ro(time[-1], c)) / (
                    2 * np.sqrt(2 * self.model.f_ro(time[-1], c)) * np.power(x[0], 2))

        if (k == 0 and m == 3) or (k == 3 and m == 0):
            return self.model.trend.d_func_beta(time[-1], x, c) * (x[1] + z0 / self.model.f_ro(time[-1], c)) / (
                    2 * np.sqrt(2 * self.model.f_ro(time[-1],  c)) * np.power(x[0], 2))

        if (k == 1 and m == 1):
            return 0

        if (k == 1 and m == 2) or (k == 2 and m == 1):
            return -0.5 * self.model.trend.d_func_gamma(time[-1], x, c) / (
                    np.sqrt(2 * self.model.f_ro(time[-1],   c)) * x[0])

        if (k == 1 and m == 3) or (k == 3 and m == 1):
            return -0.5 * self.model.trend.d_func_beta(time[-1], x, c) / (
                    np.sqrt(2 * self.model.f_ro(time[-1],  c)) * x[0])

        if (k == 2 and m == 2): 
            s1 = .5 * np.power(self.model.trend.d_func_gamma(time[-1], x, c), 2) / np.power(
                self.model.f_ro(time[-1], c), 1.5) - self.model.trend.d2_func_gamma(time[-1], x, c) / np.sqrt(
                self.model.f_ro(time[-1], c))
            s2 = z0 / self.model.f_ro(time[-1],  c) + x[1]
            return s1 * s2 / (2 * np.sqrt(2) * x[0])

        if (k == 2 and m == 3) or (k == 3 and m == 2): 

            s1 = .5 * self.model.trend.d_func_gamma(time[-1], x, c) * self.model.trend.d_func_beta(time[-1], x, c) / np.power(
                self.model.f_ro(time[-1], c), 1.5) - self.model.trend.d2_func_gamma_beta(time[-1], x, c) / np.sqrt(
                self.model.f_ro(time[-1], c))
            s2 = z0 / self.model.f_ro(time[-1],  c) + x[1]

            return s1 * s2 / (2 * np.sqrt(2) * x[0])

        if (k == 3 and m == 3):
             s1 = .5 * np.power(self.model.trend.d_func_beta(time[-1], x, c), 2) / np.power(
                self.model.f_ro(time[-1],  c), 1.5) - self.model.trend.d2_func_beta(time[-1], x, c) / np.sqrt(
                self.model.f_ro(time[-1],  c))
             s2 = z0 / self.model.f_ro(time[-1],  c) + x[1]

             return s1 * s2 / (2 * np.sqrt(2) * x[0])

    
    def d_F(self,i,k):
        return self.d_C(i,  k)*np.exp(-np.power(self.C(i), 2))/np.sqrt(np.pi)

    def d2_F(self, i, k, m):
        return (self.d2_C(i, k, m) - 2*self.C(i)*self.d_C(i, k)*self.d_C(i, m))*np.exp(-np.power(self.C(i), 2))/np.sqrt(np.pi)

    def d2_lnF(self, k, m):
        return float(sum([self.d2_F(i, k, m)/self.F(i) - self.d_F(i, k)*self.d_F(i, m)/np.power(self.F(i), 2) for i in range(len(self.time))]))

    def I11_F(self):
        return self.d2_lnF(0,0)

    def I12_F(self):
        return self.d2_lnF(0,1)

    def I13_F(self):
        return self.d2_lnF(0,2)

    def I14_F(self):
        return self.d2_lnF(0,3)

    def I22_F(self):
        return self.d2_lnF(1,1)

    def I23_F(self):
        return self.d2_lnF(1,2)

    def I24_F(self):
        return self.d2_lnF(1,3)

    def I33_F(self):
        return self.d2_lnF(2,2)

    def I34_F(self):
        return  self.d2_lnF(2,3)

    def I44_F(self):
        return self.d2_lnF(3,3)