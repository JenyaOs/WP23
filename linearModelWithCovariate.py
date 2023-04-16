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

    def d_func_b(self, b, c=0):
        return c*np.exp(b*c)

    def d2_func_b(self, b, c=0):
        return np.power(c,2)*np.exp(b*c)

class linearTrendWithCovariate():

    def __init__(self, cov=powerFuncCov()):
        self.cov_func = cov

    def get_type(self):
        return 'linearTrendWithCovariate'

    def func(self, x, t, c=0):
        return t/self.cov_func.func(x[2], c)

    def d_func_beta(self, t, x, c):
        return -self.cov_func.d_func_b(x[2], c)*t/np.power(self.cov_func.func(x[2], c), 2)

    def d2_func_beta(self, t, x, c):

        return -t*self.cov_func.d2_func_b(x[2], c)/self.cov_func.func(x[2], c)+2*t*np.power(self.cov_func.d_func_b(x[2], c), 2)/np.power(self.cov_func.func(x[2], c), 3)

x = [0.2 , 1.00 , 0.5 ]

z0 = 30
countObject = 100
M = 10000

linearModelWithCov = model.wienerModel(x, linearTrendWithCovariate(), z0)
sample = data.generateWienerLinearWithCovariate(countObject, linearModelWithCov, x, z0)
print('ОМП:', linearModelWithCov.estimateParameters(x, sample))
print('ОМП по последним наблюдениям:', linearModelWithCov.estimateConditionalParameters(x, sample))

#testingIMF.MonteCarloLinearWithCovariate(M, x, z0, linearTrendWithCovariate(), countObject)
linearModelWithCov.setParam(x)

imf = IMF.Wiener_ConditionIMF(linearModelWithCov, countObject, [0.5, 1.5], [0.5, 0.5], sample[0], z0)
matrix = np.array(imf.getIMF())
print(matrix)
print(np.linalg.det(matrix))

linearModelWithCov.setParam(x)

plan = optimalPlan.OptimalPlanningProcedure(linearModelWithCov, countObject, [0.5, 1.5], [0.5, 0.5], sample[0][0])
print(plan.directSearchD())
