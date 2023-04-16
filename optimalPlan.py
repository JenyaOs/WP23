import IMF
import numpy as np
import scipy as sp

class OptimalPlanningProcedure():
    def __init__(self, _model, _n, _cov0, _weight0, _time0):
        self.covariate = _cov0
        self.weight = _weight0
        self.time = _time0
        self.model = _model
        self.n = _n
        self.z = self.model.z

    def getIMF_D(self):
        imf = IMF.Wiener_IMF(self.model, self.n, self.covariate, self.weight, [self.time for i in range(self.n)])
        matrix = np.array(imf.getIMF())
        return np.linalg.det(matrix)

    def getConditionIMF_D(self):
        imf = IMF.Wiener_ConditionIMF(self.model, self.n, self.covariate, self.weight, [self.time for i in range(self.n)], self.z)
        matrix = np.array(imf.getIMF())
        return np.linalg.det(matrix)

    def getLoads(self, cov):
        self.covariate = cov
        return -self.getConditionIMF_D()

    def getWeight(self):
        new_weight = [0.,1.]
        x = 0
        step = 1.0/self.n
        max = 0
        round_e = len(str(self.n))-1
        for i in range(int(self.n)):
            weight_x = [round(x,round_e),round(1 - x,round_e)]
            self.weight = weight_x
            new_max = self.getConditionIMF_D()
            if(new_max>max):
                max = new_max
                new_weight = weight_x
            x += step

        return new_weight

    def getFireTime(self, time):
        fire = 0
        if(time[0]<0):
            fire+=np.power(time[0],2)

        for i in range(len(time)-1):
            if(time[i]>time[i+1]):
                fire += np.power(time[i]-time[i+1],2)

        return  fire

    def getTime(self, time):
        print(time)
        self.time =  time
        return -self.getIMF_D()

    def directSearchD(self):
        IMF_0,conditionalIMF_0 = self.getIMF_D(),self.getConditionIMF_D()
        IMF_n,conditionalIMF_n = 0,0

        print("Начальный план", self.getPlan())

        while (abs(IMF_0 - IMF_n) > 0.01 and abs(conditionalIMF_0 - conditionalIMF_n) > 0.01):
            IMF_n, conditionalIMF_n = IMF_0,conditionalIMF_0

            #Зафиксировать время и веса, запустить оптимизацию по ковариатам
            cov_n = sp.optimize.minimize(self.getLoads, self.covariate,
                                         bounds=((.1, 5), (0.1, 5)),
                                         method='TNC', jac=None, tol=None, callback=None)
            self.covariate = cov_n.x
            #print(self.covariate)

            weight_n = self.getWeight()
            self.weight = weight_n
            print(self.weight)

            fireTime = ({"type": 'eq', "fun": self.getFireTime})
            time_n = sp.optimize.minimize(self.getTime, self.time, constraints=fireTime,
                                            #bounds=((.1, 5), (0.1, 5)),
                                            method='SLSQP', jac=None, tol=None, callback=None)
            self.time = time_n.x


            IMF_0, conditionalIMF_0 =  self.getIMF_D(),self.getConditionIMF_D()

        print("Оптимальный план", self.getPlan())
        return  IMF_0, conditionalIMF_0

    def getPlan(self):
        return self.covariate, self.weight, self.time