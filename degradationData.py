import numpy as np
import wienerModel as model
import matplotlib.pyplot as plt

class experiment():
    def __init__(self, _countObject, _time=[[]], _value = [[]], _loads=[[]]):
        self.countObject = _countObject
        self.time = _time
        self.value = _value
        self.loads = _loads

    def generateExperiment(self):
        return

    def readExperimentFromFile(self):
        return

class degradationData():

    def __init__(self, time, delta, value, covariates=[]):
        self.time = time
        self.delta = delta
        self.value = value
        self.covariates = covariates

    def convertDataFromExperiment(self):
        return 0

class GenerateWienerData():

    def __init__(self, _model):
        self.model = _model
        # generation moments measurement of degradation value on interval with fix step

    def generatortime(self, _len, _countIntervals):
        """
        :param _len: interval of measurement
        :param _countIntervals: number of parts which divided
        :return: array of time
        """
        return np.linspace(0, _len, _countIntervals)
        # generation values of degradation processes

    def generator_trend(self, time, x, c=0):
        """
        :param time: the time of the experiment
        :param x: the parameter of model
        :param c: the value of covariates vector
        :return: array of value for building function trend
        """
        return [self.model.function_trend(time[i], x, c) for i in range(len(time))]

        #  generation increments of degradation processes

    def get_Delta(self, time, x, c=0):
        """
        :param time: the time of the experiment
        :param x: the parameter of model
        :param c: the value of covariates vector
        :return: array of increments of degradation processes
        """
        arrayD = np.zeros(len(time) - 1)

        for i in range(len(time) - 1):
            # increments between time moments
            delta = self.model.f_ro(time[i + 1], c) - self.model.f_ro(time[i], c)
            arrayD[i]=(self.model.increment(delta))

        return arrayD

    def get_values_limited(self, delta, time, z0):
        value = []
        times = []
        for i in range(len(delta)):
            value.append([0])
            times.append(time[i])
            for j in range(len(delta[i])):
                if (value[i][j]+delta[i][j] <= z0):
                    value[i].append(value[i][j]+delta[i][j])
                else:
                    times[i] = (time[i][0:j + 1])
                    delta[i] = (delta[i][0:j + 1])
                    break
        return value, delta, times

    def get_values_unlimited(self, delta, time, z0):
        value = []
        times = []
        for i in range(len(delta)):
            value.append([0])
            times.append(time[i])
            for j in range(len(delta[i])):
                value[i].append(value[i][j] + delta[i][j])

        return value, delta, times

    def show_data(self, data, c = 0):
        time = data[0]
        value = data[2]
        cov = data[3]
        if (c == 0):
            for i in range(len(time)):
                plt.plot(time[i],  value[i], 'grey')

        else:
            for i in range(len( time)):
                if (cov[i] == c):
                    plt.plot( time[i],  value[i], 'grey')


        plt.plot(np.linspace(0, max([max(time[i]) for i in range(len(time))]), 100), [self.model.z for i in range(100)], 'red')
        plt.show()
        return

def generateWienerLinear(M, model, x, z0):
    model = GenerateWienerData(model)
    time = []
    covariate = [0 for i in range(M)]

    for i in range(M):
        time.append(np.linspace(0, 40, 100))

    delta = [model.get_Delta(time[i], x, covariate[i]) for i in range(M)]
    value, delta, time = model.get_values_unlimited(delta, time, z0)
    model.show_data([time, delta, value, covariate], 0)
    return [time, delta, value, covariate]

def generateWienerLinearWithCovariate(M, model, x, z0):
    model = GenerateWienerData(model)
    time = []

    covariate = [(i%2)*1.5+0.5 for i in range(M)]

    for i in range(M):
            time.append(np.linspace(0, 25*np.exp(min(covariate)*x[2])+2, 20))

    delta = [model.get_Delta(time[i], x, covariate[i]) for i in range(M)]
    value, delta, time = model.get_values_unlimited(delta, time, z0)
    model.show_data([time, delta, value, covariate], 0)
    return [time, delta, value, covariate]

def generateWienerPowerWithCovariate(M, model, x, z0):
    model = GenerateWienerData(model)
    time = []

    covariate = [(i%2)*1.5+0.5 for i in range(M)]

    for i in range(M):
            time.append(np.linspace(0, np.power(25*np.exp(min(covariate)*x[2]*x[3]),1/x[2])+2, 20))

    delta = [model.get_Delta(time[i], x, covariate[i]) for i in range(M)]
    value, delta, time = model.get_values_unlimited(delta, time, z0)
    model.show_data([time, delta, value, covariate], 0)
    return [time, delta, value, covariate]
