import numpy as np
from numpy import inf, dot
from decimal import Decimal as dc

from decimal import *
global nfevalscontrol
getcontext().prec = 30

class rko64:
  def __init__(self):

    self.c2 = dc(2)/dc(15)
    self.c3 = dc(1)/dc(3)
    self.c4 = dc(1)/dc(3)
    self.c5 = dc(2)/dc(3)
    self.c6 = dc(7)/dc(9)

    self.a21 = dc(2)/dc(15)

    self.a31 = dc(1)/dc(20)
    self.a32 = dc(3)/dc(20)

    self.a41 = dc(11)/dc(108)
    self.a42 = -dc(5)/dc(36)
    self.a43 = dc(10)/dc(27)

    self.a51 = dc(23)/dc(54)
    self.a52 = -dc(5)/dc(18)
    self.a53 = -dc(35)/dc(54)
    self.a54 = dc(7)/dc(6)

    self.a61 = -dc(119)/dc(324)
    self.a62 = dc(385)/dc(972)
    self.a63 = dc(260)/dc(243)
    self.a64 = -dc(182)/dc(243)
    self.a65 = dc(104)/dc(243)

    self.a71 = dc(1067)/dc(2044)
    self.a72 = -dc(105)/dc(292) 
    self.a73 = -dc(5830)/dc(6643)
    self.a74 = dc(108)/dc(73)
    self.a75 = -dc(216)/dc(511)
    self.a76 = dc(4374)/dc(6643)

    self.a81 = dc(31)/dc(420)
    self.a82 = dc(0)
    self.a83 = dc(3125)/dc(17472)
    self.a84 = dc(81)/dc(320)
    self.a85 = dc(27)/dc(140)
    self.a86 = dc(6561)/dc(29120)
    self.a87 = dc(73)/dc(960)



    self.b1 = -dc(1)/dc(24)
    self.b2 = dc(0)
    self.b3 = dc(125)/dc(168)
    self.b4 = -dc(3)/dc(8)
    self.b5 = dc(33)/dc(56)
    self.b6 = dc(0)
    self.b7 = dc(0)
    self.b8 = dc(1)/dc(12)
    self.fevals = 7
  
  def step(self,ode, t0, y0, h, K1):
      k1 = h * K1
      k2 = h * ode.fullEval(t0 + h * self.c2, y0 + k1 * self.a21)
      k3 = h * ode.fullEval(t0 + h * self.c3, y0 + self.a31 * k1 + self.a32 * k2)
      k4 = h * ode.fullEval(t0 + h * self.c4, y0 + self.a41 * k1 + self.a42 * k2 + self.a43 * k3)
      k5 = h * ode.fullEval(t0 + h * self.c5, y0 + self.a51 * k1 + self.a52 * k2 + self.a53 * k3 + self.a54 * k4)
      k6 = h * ode.fullEval(t0 + h * self.c6, y0  + self.a61 * k1 + self.a62 * k2 + self.a63 * k3 + self.a64 * k4 + self.a65 * k5)
      k7 = h * ode.fullEval(t0 + h, y0 + self.a71 * k1 + self.a72 * k2 + self.a73 * k3 + self.a74 * k4 + self.a75 * k5 + self.a76 * k6)
      
      
      y4 = y0 + self.a81 * k1 + self.a83 * k3 + self.a84 * k4 + self.a85 * k5 + self.a86 * k6 + self.a87 * k7
      k8 =  ode.fullEval(t0 + h, y4)
      
      y6 = y0 + self.b1 * k1 + self.b3 * k3 + self.b4 * k4 + self.b5 * k5 + h*self.b8 * k8
      err = y6-y4
      return y6, err, k8

class rkb64:
  def __init__(self):
   
  
    self.A11 = np.array([
                         
                         [dc(1)/dc(9), dc(1)/dc(9), dc(0),  dc(0),  dc(0),  dc(0),dc(0)],
                         [dc(1)/dc(12), dc(0), dc(1)/dc(12), dc(0), dc(0), dc(0), dc(0)],
                         [-dc(1)/dc(44), dc(0) , dc(9)/dc(22), dc(5)/dc(44), dc(0), dc(0),dc(0)],
                         [dc(7)/dc(36), dc(0), dc(0), dc(5)/dc(9), dc(1)/dc(12), dc(0), dc(0)],
                         [-dc(3)/dc(7), dc(0), dc(9)/dc(8), -dc(5)/dc(28), dc(27)/dc(56), dc(0), dc(0)],
                         [dc(7)/dc(150), dc(0), dc(27)/dc(100), dc(11)/dc(30), dc(27)/dc(100), dc(7)/dc(150), dc(0)]
    ])
    self.A12 = np.array([
                         
                         [dc(2)/dc(9), dc(0), dc(0), dc(0),dc(0), dc(0), dc(0)],
                         [dc(5)/dc(48), dc(1)/dc(16), dc(0), dc(0) , dc(0), dc(0), dc(0)],
                         [dc(37)/dc(176), dc(243)/dc(176) , -dc(12)/dc(11), dc(0), dc(0), dc(0), dc(0)],
                         [-dc(635)/dc(432), -dc(167)/dc(16), dc(100)/dc(9), dc(44)/dc(27), dc(0), dc(0), dc(0)],
                         [-dc(29)/dc(4), dc(1377)/dc(28), -dc(1425)/dc(28), -dc(11)/dc(2), dc(27)/dc(28),dc(0), dc(0)],
                          [dc(7)/dc(150), dc(0), dc(27)/dc(100), dc(11)/dc(30), dc(27)/dc(100), dc(7)/dc(150), dc(0)]
    ])
    self.A21 = np.array([
                         
                         [dc(1)/dc(9), dc(1)/dc(9), dc(0), dc(0), dc(0), dc(0), dc(0)],
                         [dc(7)/dc(48), dc(3)/dc(16), -dc(1)/dc(6), dc(0), dc(0), dc(0), dc(0)],
                         [-dc(31)/dc(176), -dc(81)/dc(176), dc(45)/dc(44), dc(5)/dc(44), dc(0), dc(0),dc(0)],
                         [dc(73)/dc(144), dc(15)/dc(16), -dc(5)/dc(4), dc(5)/dc(9), dc(1)/dc(12), dc(0), dc(0)],
                         [-dc(39)/dc(28), -dc(81)/dc(28), dc(279)/dc(56), -dc(5)/dc(28), dc(27)/dc(56), dc(0), dc(0)],
                         [dc(7)/dc(150), dc(0), dc(27)/dc(100), dc(11)/dc(30), dc(27)/dc(100), dc(7)/dc(150), dc(0)]
    ])
    self.A22 = np.array([
                         
                         [dc(1)/dc(9), dc(1)/dc(9), dc(0),dc(0), dc(0), dc(0), dc(0)],
                         [dc(7)/dc(48), dc(3)/dc(16), -dc(1)/dc(6), dc(0),dc(0), dc(0), dc(0)],
                         [-dc(185)/dc(1584), -dc(123)/dc(880), dc(2)/dc(3), dc(89)/dc(990), dc(0), dc(0),dc(0)],
                         [dc(1031)/dc(3888), -dc(53)/dc(144), dc(65)/dc(324), dc(317)/dc(486), dc(1)/dc(12), dc(0), dc(0)],
                         [-dc(29)/dc(63), dc(15)/dc(7), -dc(103)/dc(168), -dc(139)/dc(252), dc(27)/dc(56), 0, 0],
                         [dc(7)/dc(150), dc(0), dc(27)/dc(100), dc(11)/dc(30), dc(27)/dc(100), dc(7)/dc(150), 0]
    ])
    self.B = np.array([dc(7)/dc(150), dc(0), dc(27)/dc(100), dc(11)/dc(30), dc(27)/dc(100), dc(7)/dc(150),dc(0)])
    self.D = np.array([dc(13)/dc(200), dc(0), dc(183)/dc(800), dc(33)/dc(80), dc(183)/dc(800), dc(7)/dc(300), dc(1)/dc(24)])
    self.E = (self.D - self.B)
    self.C = np.array([dc(0), dc(2)/dc(9), dc(1)/dc(6), dc(1)/dc(2), dc(5)/dc(6), dc(1), dc(1)])
  def step(self, ode, t0, y0, h, K1):
    l = ode.l
    size = ode.size
    k = np.array([[dc(0) for i in range(7)] for l in range(size)])
    yt = np.array(range(size))
    k[:, 0] = K1
  
    print(k)
    for i in range(1,7):
   
      for j in range(l):
        for p in range(l):
          
          yt[p] = y0[p] + h * dot(k[p], self.A11[i-1])
        for p in range(l, size):
          yt[p] = y0[p] + h * dot(k[p], self.A12[i-1])
        k[j, i] = ode.partEval(j)(t0 + h * self.C[i-1], yt)
      for j in range(l, size):
        for p in range(l):
          yt[p] = y0[p] + h * dot(k[p], self.A21[i-1])
        for p in range(l, size):
          yt[p] = y0[p] + h * dot(k[p], self.A22[i-1])
        k[j, i] = ode.partEval(j)(t0 + h * self.C[i-1], yt)
    y6 = np.copy(y0)
    y4 = np.copy(y0)
    for i in range(size):
      y6[i] = y0[i] + h * dot(k[i], self.D)
      y4[i] = y0[i] + h * dot(k[i], self.B)
   
    
    return y6, y6 - y4, k[:,6]
      

def rkAutostep(ode, t0, y0, tfin, hmax, tol, method, order):
    global nfevalscontrol
    nfevalscontrol = 0
    h = 2
    y = y0
    T = [t0]
    Y = [y0]
    stats = {
        'nsteps' : 0,
        'nrejected' : 0,
        'nfevals' : 0,
        'steps' : [0],
        'err' : [0],
        'y' : None,
        't' : None
    }
    facmin = dc('0.1')
    facmax = dc(5)
    factor = dc('0.8')
  
    errNorm = tol
    _k1 = ode.fullEval(t0, y0)
    d1 = max(np.linalg.norm(_k1, inf), dc(1e-5))
    d0 = max(np.linalg.norm(y0, inf),dc(1e-5))
    h0 = dc(1) / dc(100) * d0 / d1
    y1 = y0 + h0 * _k1
    d2 = np.linalg.norm(ode.fullEval(t0 + h0, y1) - _k1, inf) / h0
    h1 = dc(1) / dc(100) / max(d1, d2)
    
    if max(d1, d2) <dc(1e-15):
      h1 = max(dc(1e-6), h0 * dc(1)/dc(1000))
    else:
      h1 = h1 **( dc(1)/(order + dc( 1)))
    h = min(h1, dc(100) * h0)

    pw = dc(1)/(order + dc(1))
    
    nofailed = True
    
    _k1 = ode.fullEval(t0,y0)
    while T[-1] < tfin :
      if T[-1] + h > tfin:
        h = tfin - T[-1]
      
    
      if not nofailed:
        _k1 = ode.fullEval(T[-1], Y[-1])
        stats['nfevals'] = stats['nfevals']+1
      y, err, k1 = method(ode,  T[-1],  Y[-1],  h, _k1)
      _k1 = k1
      errNorm = (np.linalg.norm(err, inf))/np.linalg.norm(y,inf)
      h = h *  min(facmax, max(facmin,  factor *( (( tol/errNorm) **( pw) ) )))
      stats['nfevals'] = stats['nfevals']+7
      if  errNorm > tol:
        
        nofailed = False
        stats['nrejected'] +=1
        continue
      
      
      T.append(T[-1] + h)
      Y.append(y )
      stats['nsteps'] += 1
      stats['steps'].append(h)
      stats['err'].append(errNorm)
      

    stats['y'] = np.array(Y).T.tolist()
    stats['t'] = np.array(T).tolist()
    return np.array(T), np.array(Y).T, stats


mu1 = dc('0.012277471')
mu2 = dc(1) - mu1




def f1(t, y):
  return y[3]
def f2(t, y):
  return y[2] - 2 * y[3] - mu2 * (y[2]/ (((y[0] + mu1)**2 + y[2]**2)**3).sqrt()) - mu1 * (y[2]/ (((y[0] - mu2)**2 + y[2]**2)**3).sqrt())
def f3(t, y):
  return y[1]
def f4(t, y):
  return y[0] + 2 * y[1] - mu2 * ((y[0] + mu1)/ (((y[0] + mu1)**2 + y[2]**2)**3).sqrt()) - mu1 * ((y[0] - mu2)/ (((y[0] - mu2)**2 + y[2]**2)**3).sqrt())

class ODE:
  def __init__(self, parts, l):
    self.parts = parts
    self.l = l
    self.size = len(parts)
  def partEval(self, i):
    return self.parts[i]
  def fullEval(self, t, y):
    return np.array([
                   f(t, y) for f in self.parts
  ])
ode = ODE([f1, f2, f3, f4], 2)