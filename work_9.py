import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import objective 


# №1. Даны значения величины заработной платы заемщиков банка (zp) и значения 
# их поведенческого кредитного скоринга (ks): 
# zp = [35, 45, 190, 200, 40, 70, 54, 150, 120, 110], 
# ks = [401, 574, 874, 919, 459, 739, 653, 902, 746, 832]. 
# Используя математические операции, посчитать коэффициенты линейной регрессии, 
# приняв за X заработную плату (то есть, zp - признак), 
# а за y - значения скорингового балла (то есть, ks - целевая переменная). 
# Произвести расчет как с использованием intercept, так и без.

zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])
x = zp
y = ks
n = len(x)
# plt.scatter(x, y)
# plt.show()

b1 = (n*np.sum(x*y) - np.sum(x)*np.sum(y))/ (n*np.sum(x**2)-np.sum(x)**2)
print(b1)
b0 = np.mean(y) - b1*np.mean(x)
print(b0)

y_pred = b0 + b1 * x
print('расчитанный y= ', y_pred)
resid = y - y_pred
print(stats.shapiro(resid)) # Норм

mse = np.sum((y - y_pred)**2) / n  # почему то число дикое получается?
print('ошибка =', mse)

model = LinearRegression()
x = x.reshape(-1, 1)
regres = model.fit(x, y)
print(regres.coef_)
print(regres.intercept_)

plt.scatter(x, y)
plt.plot(x, b1 * x + b0, 'r')
plt.show()

# x = sm.add_constant(x)
# model = sm.OLS(y, x)
# res = model.fit()
# print(res.summary())


#################################################################################
#################################################################################

# Задача 2 Посчитать коэффициент линейной регрессии при заработной плате (zp), используя
# градиентный спуск (без intercept).

a = 10**-10
B1 = 0.1

for i in range(30):
    B1-= a * (2/n) * np.sum((B1*x-y)*x)
    # if i%100==0:
    print('B1 ={}'.format(B1))



####################################################################################
####################################################################################

# Задача 3 (Дополнительно) Произвести вычисления как в пункте 2, но с вычислением intercept. Учесть, что
# изменение коэффициентов должно производиться
# на каждом шаге одновременно (то есть изменение одного коэффициента не должно
# влиять на изменение другого во время одной итерации).
# B1 = (2/n) * np.sum((b1*x-y)*x)
# print(B1)

# new_B1 = B1 - 0.1 *B1#(2/n) * np.sum((b1*x-y)*x) 
# print(new_B1)

solution = []
evaluation = []

for i in range(20):
    solution.append(B1)
    B1_evaluation = np.object_(B1)
    evaluation.append(B1_evaluation)

    gradient = (2/n) * np.sum((B1*x-y)*x)
    new_B1 = B1 - 0.1* gradient
    B1 = new_B1
    print(i, '',  B1,'',   B1_evaluation)

