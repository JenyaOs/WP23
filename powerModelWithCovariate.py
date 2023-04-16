import degradationData as data
import wienerModel as model
import numpy as np
import testingIMF
import IMF
import optimalPlan

class powerFuncCov():

    def get_type(self):
        return powerFuncCov

    def func(self, b, c=0):
        return np.exp(b*c)

    def d_func_b(self, g, b, c=0):
        return c*g*np.exp(g*b*c)

    def d2_func_b(self, g, b, c=0):
        return np.power(c*g,2)*np.exp(g*b*c)

class powerTrendWithCovariate():

    def __init__(self, cov=powerFuncCov()):
        self.cov_func = cov

    def get_type(self):
        return 'powerTrendWithCovariate'

    def func(self, x, t, c=0):
        return np.power(t/self.cov_func.func(x[3], c), x[2])

    def d_func_beta(self, t, x, c):
        return -self.cov_func.d_func_b(x[2], x[3], c)*np.power(t,x[2])/np.power(self.cov_func.func(x[3], c), 2)

    def d2_func_beta(self, t, x, c):
        return -np.power(t,x[2])*self.cov_func.d2_func_b(x[2], x[3], c)/self.cov_func.func(x[3], c)+2*t*np.power(self.cov_func.d_func_b(x[2],x[3], c), 2)/np.power(self.cov_func.func(x[3], c), 3)

    def d_func_gamma(self, t, x, c):
        return self.func(x, t, c)*np.log(t/self.cov_func.func(x[3],c))

    def d2_func_gamma(self, t, x, c):
        return self.func(x, t, c)*np.power(np.log(t/self.cov_func.func(x[3],c)),2)

    def d2_func_gamma_beta(self, t, x, c):
        return c*self.func(x,t,c)*(1-x[2]*(np.log(t)-c*x[3]))

x = [0.2 , 1.00 , 1.1, 0.5 ]

z0 = 30
countObject = 100
M = 10000

powerModelWithCov = model.wienerModel(x, powerTrendWithCovariate(), z0)
sample = data.generateWienerPowerWithCovariate(countObject, powerModelWithCov, x, z0)
print('ОМП:', powerModelWithCov.estimateParameters(x, sample))
print('ОМП по последним наблюдениям:', powerModelWithCov.estimateConditionalParameters(x, sample))

#testingIMF.MonteCarloLinearWithCovariate(M, x, z0, linearTrendWithCovariate(), countObject)
powerModelWithCov.setParam(x)

imf = IMF.Wiener_ConditionIMF(powerModelWithCov, countObject, [0.5, 1.5], [0.3, 0.7], sample[0], z0)
matrix = np.array(imf.getIMF())
print(matrix)
print(np.linalg.det(matrix))

powerModelWithCov.setParam(x)

plan = optimalPlan.OptimalPlanningProcedure(powerModelWithCov, countObject, [0.5, 1.5], [0.3, 0.7], sample[0][0])
print(plan.directSearchD())

