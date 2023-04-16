import degradationData as sample
import wienerModel as model
import numpy as np

def getEstimatedIMF(x_mean, x_est, matrix = [[1, 1], [1, 1]]):
    print(len(x_est[0]))
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = sum([(x_est[i][k] - x_mean[i]) * (x_est[j][k] - x_mean[j]) for k in range(len(x_est[j]))]) \
                        / len(x_est[j])
    return matrix

def MonteCarloLinearWithCovariate(M, x, z0, trend, countObject):
    estimate_x = []
    for i in range(M):
        linearModel = model.wienerModel(x, trend, z0)
        data = sample.generateWienerLinearWithCovariate(countObject, linearModel, x, z0)
        estimate_x.append(linearModel.estimateConditionalParameters(x, data))
        print(estimate_x[i])

    link = np.transpose(estimate_x)
    x_mean = [np.average(link[0]),np.average(link[1]),np.average(link[2])]
    print(x_mean)

    cov_matrix =  getEstimatedIMF(x_mean, link, [[1,1,1],[1,1,1],[1,1,1]])
    inf_matrix = np.linalg.inv(cov_matrix)

    #print("Ковариационная матрица: ", cov_matrix)
    #print("Определитель ков . матрицы: ", np.linalg.det(cov_matrix))

    print("ИМФ: ", inf_matrix)
    print("Определитель ИМФ", np.linalg.det(inf_matrix))

    return link

def MonteCarloLinear(M, x, z0, trend, countObject):
    estimate_x = []
    for i in range(M):
        linearModel = model.wienerModel(x, trend, z0)
        data = sample.generateWienerLinear(countObject, linearModel, x, z0)

        estimate_x.append(linearModel.estimateParameters(x, data))
        print(estimate_x[i])

    link = np.transpose(estimate_x)
    x_mean = [np.average(link[0]),np.average(link[1])]
    print(x_mean)

    cov_matrix =  getEstimatedIMF(x_mean, link, [[1,1],[1,1]])
    inf_matrix = np.linalg.inv(cov_matrix)

    print("ИМФ: ", inf_matrix)
    print("Определитель ИМФ", np.linalg.det(inf_matrix))

    return
