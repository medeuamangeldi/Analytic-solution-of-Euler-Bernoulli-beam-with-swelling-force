!pip install gekko
from gekko import GEKKO

import numpy as np
import matplotlib.pyplot as plt

yData = np.array([0.00000000e+00, 7.90602967e-06, 3.14944931e-05, 7.05709520e-05, 1.24940968e-04, 1.94410103e-04, 2.78783918e-04, 3.77867975e-04, 4.91467835e-04, 6.19389061e-04, 7.61437214e-04, 9.17417856e-04, 1.08713655e-03, 1.27039885e-03, 1.46701033e-03, 1.67677654e-03,
 1.89950305e-03, 2.13499542e-03, 2.38305921e-03, 2.64349998e-03,
 2.91612329e-03, 3.20073471e-03, 3.49713979e-03, 3.80514411e-03,
 4.12455321e-03, 4.45517267e-03, 4.79680803e-03, 5.14926488e-03,
 5.51234876e-03, 5.88586524e-03, 6.26961987e-03, 6.66341823e-03,
 7.06706588e-03, 7.48036837e-03, 7.90313126e-03, 8.33516012e-03,
 8.77626052e-03, 9.22623800e-03, 9.68489814e-03, 1.01520465e-02,
 1.06274886e-02, 1.11110301e-02, 1.16024764e-02, 1.21016333e-02,
 1.26083061e-02, 1.31223006e-02, 1.36434221e-02, 1.41714764e-02,
 1.47062690e-02, 1.52476054e-02, 1.57952912e-02, 1.63491319e-02,
 1.69089332e-02, 1.74745005e-02, 1.80456395e-02, 1.86221556e-02,
 1.92038546e-02, 1.97905418e-02, 2.03820230e-02, 2.09781036e-02,
 2.15785892e-02, 2.21832853e-02, 2.27919977e-02, 2.34045317e-02,
 2.40206929e-02, 2.46402870e-02, 2.52631194e-02, 2.58889958e-02,
 2.65177217e-02, 2.71491027e-02, 2.77829443e-02, 2.84190521e-02,
 2.90572316e-02, 2.96972884e-02, 3.03390281e-02, 3.09822563e-02,
 3.16267784e-02, 3.22724001e-02, 3.29189269e-02, 3.35661644e-02,
 3.42139181e-02, 3.48619937e-02, 3.55101966e-02, 3.61583324e-02,
 3.68062067e-02, 3.74536251e-02, 3.81003931e-02, 3.87463163e-02,
 3.93912001e-02, 4.00348503e-02, 4.06770724e-02, 4.13176719e-02,
 4.19564543e-02, 4.25932253e-02, 4.32277904e-02, 4.38599552e-02,
 4.44895252e-02, 4.51163060e-02, 4.57401031e-02, 4.63607222e-02,
 4.69779687e-02])
xData = np.array([0,     0.0254, 0.0508, 0.0762, 0.1016, 0.127,  0.1524, 0.1778, 0.2032, 0.2286,
 0.254,  0.2794, 0.3048, 0.3302, 0.3556, 0.381,  0.4064, 0.4318, 0.4572, 0.4826,
 0.508,  0.5334, 0.5588, 0.5842, 0.6096, 0.635,  0.6604, 0.6858, 0.7112, 0.7366,
 0.762,  0.7874, 0.8128, 0.8382, 0.8636, 0.889,  0.9144, 0.9398, 0.9652, 0.9906,
 1.016,  1.0414, 1.0668, 1.0922, 1.1176, 1.143,  1.1684, 1.1938, 1.2192, 1.2446,
 1.27,  1.2954, 1.3208, 1.3462, 1.3716, 1.397,  1.4224, 1.4478, 1.4732, 1.4986,
 1.524,  1.5494, 1.5748, 1.6002, 1.6256, 1.651,  1.6764, 1.7018, 1.7272, 1.7526,
 1.778,  1.8034, 1.8288, 1.8542, 1.8796, 1.905,  1.9304, 1.9558, 1.9812, 2.0066,
 2.032,  2.0574, 2.0828, 2.1082, 2.1336, 2.159,  2.1844, 2.2098, 2.2352, 2.2606,
 2.286,  2.3114, 2.3368, 2.3622, 2.3876, 2.413,  2.4384, 2.4638, 2.4892, 2.5146,
 2.54])

#s = 76.2
#sigma = 1000000
#E = 20.76
#I = 0.00249739
#c = 0.03
#d = 4.572
#K = s*sigma/(E*I)
#B = np.log(10)/(c*d)

# measurements
xm = xData
ym = yData

# GEKKO model
m = GEKKO()

# parameters
x = m.Param(value=xm)

#a = m.FV(lb=0.001,ub=500)
#b = m.FV(lb=0.001,ub=4000)
#c = m.FV(lb=0.001,ub=2000)
#d = m.FV(lb=0.001,ub=100)
#e = m.FV(lb=0.001,ub=20)

a = m.FV(0.01)
b = m.FV(0.01)

a.STATUS=1
b.STATUS=1

# variables
y = m.CV(value=ym)

y.FSTATUS=1

# regression equation
m.Equation(y==1/a*m.log(a*x+m.exp(b))-b/a)

# regression mode
m.options.IMODE = 2
m.options.SOLVER = 1
# optimize
m.solve(disp=True, debug=0)

# print parameters
print('Optimized, a = %.5f, b = %.5f' %(a.value[0], b.value[0]))
a = a.value[0]
b = b.value[0]

absError = y.value - yData
SE = np.square(absError) # squared errors
MSE = np.mean(SE) # mean squared errors
RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
Rsquared = 1.0 - (np.var(absError) / np.var(yData))
print('RMSE:', RMSE)
print('R-squared:', Rsquared)

plt.plot(xm,ym, color='black', label = 'FE solutions')
plt.plot(xm,y.value, label = 'fit')
plt.xlabel('horizontal position x (m)')
plt.ylabel('beam deformation u (m)')
plt.legend(title = 'R-squared: %.3f' %(Rsquared))
plt.grid()
plt.show()

m = GEKKO(remote=False)

# nonlinear regression
a,b = m.Array(m.FV,2,value=0.01,lb=-100,ub=100)
x = m.MV(xData); u = m.CV(yData)
a.STATUS=1; b.STATUS=1; u.FSTATUS=1
m.Equation(u==1/a*m.log(a*x+m.exp(b))-b/a)

m.options.IMODE = 2; m.options.EV_TYPE = 2
m.solve()

#stats (from other answer)
absError = u.value - yData
SE = np.square(absError) # squared errors
MSE = np.mean(SE) # mean squared errors
RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
Rsquared = 1.0 - (np.var(absError) / np.var(yData))
print('RMSE:', RMSE)
print('R-squared:', Rsquared)
print('Parameters', a.value[0], b.value[0])

# deep learning
#from gekko import brain
#b = brain.Brain()
#b.input_layer(1)
#b.layer(linear=1)
#b.layer(tanh=2)
#b.layer(linear=1)
#b.output_layer(1)
#b.learn(xData,yData,obj=1,disp=False) # train
#xp = np.linspace(min(xData),max(xData),100)
#w = b.think(xp) # predict

plt.plot(xData,yData,'k.',label='Finite Element Solution')
plt.plot(x.value,u.value,'r:',lw=3,label=r'$u=\frac{1}{a}*ln(a*x+e^b)-\frac{b}{a}$')
#plt.plot(x.value,z.value,'g--',label='c-spline')
#plt.plot(xp,w[0],'b-.',label='deep learning')
plt.legend(); plt.show()

u_l = lambda x: 1/-30.100006864*np.log(-30.100006864*x+np.exp(4.5865732043))-4.5865732043/(-30.100006864)

x = np.linspace(0,2.54,101)
plt.plot(x,u_l(x), label = 'Fit')
plt.plot(x,yData,'--', color = 'red', label = 'Finite Element solution')
plt.xlabel('x (m)')
plt.ylabel('beam deformation (m)')
plt.grid()
plt.legend()
plt.show()

def error_def(u):
  x = np.linspace(0,2.54,len(yData))
  error = []
  for i in range(len(yData)):
    error.append(abs(yData[i] - u(x)[i]))
  plt.plot(x,error)
  plt.xlabel('x (m)')
  plt.ylabel('Error')
  plt.grid()
  #plt.legend()
  plt.show()

error_def(u_l)

