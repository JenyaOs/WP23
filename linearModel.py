import degradationData as data
import wienerModel as model
import numpy as np
import testingIMF
import IMF

class linearTrend():

    def get_type(self):
        return 'linearTrend'

    def func(self, x, t, c=0):
        return t



x = [.2,1]
z0 = 50
countObject = 100
M = 1000
linearModel = model.wienerModel(x, linearTrend(), z0)
sample = data.generateWienerLinear(100, linearModel, x, z0)

print('ОМП:', linearModel.estimateParameters(x, sample))
print('ОМП по последним наблюдениям:', linearModel.estimateConditionalParameters(x, sample))

testingIMF.MonteCarloLinear(M, x, z0, linearTrend(), countObject)

imf = IMF.Wiener_IMF(linearModel, countObject, [0], 1, sample[0])
print(imf.getIMF())