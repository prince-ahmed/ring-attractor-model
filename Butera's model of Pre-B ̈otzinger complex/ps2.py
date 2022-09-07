
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class Butera():
    def __init__(self, tmax, el,reduction,phase,nullcline):
        self.nullcline=nullcline
        self.phase=phase
        self.reduction=reduction
        self.tmax = tmax
        self.el  = el

        self.cm = 21.0
        self.gna = 28.0
        if reduction:self.gna=0

        self.gk  =  11.2
        self.gl = 2.8
        self.ena =  50.0
        self.ek  = -85.0
        self.el  = el
        self.t = np.arange(0.0, tmax, 0.01)
        self.i = 0.0
        self.taubar = 10000.0
        self.gnap = 2.8

    def xinf(self, v, vt, sig):
        return 1/(1+np.exp((v-vt)/sig))
    
    def taux(self, v, vt, sig, tau):
        return tau/np.cosh((v-vt)/(2*sig))
    
    def il(self, v):
        return self.gl * (v - self.el)
    
    def minf(self, v):
        return self.xinf(v, -34.0, -5.0)

    def ina(self, v, n):
        return self.gna * (self.minf(v) ** 3) * (1-n) * (v-self.ena)
    
    def ninf(self, v):
        return self.xinf(v, -29.0, -4.0)

    def taun(self, v):
        return self.taux(v, -29.0, -4.0, 10.0)

    def ik(self, v, n):
        return self.gk * (n**4) * (v-self.ek)
    
    def mninf(self, v):
        return self.xinf(v, -40.0, -6.0)

    def hinf(self, v):
        return self.xinf(v, -48.0, 6.0)

    def tauh(self, v):
        return self.taux(v, -48.0, 60.0, self.taubar)
    
    def inap(self, v, h):
        return self.gnap * self.mninf(v) * h * (v-self.ena)

    @staticmethod
    def dALLdt(X, t, self):
        v, n, h = X
        if self.reduction: n = self.ninf(v)
        dvdt = (self.i - self.il(v) - self.ina(v, n) - self.ik(v, n) - self.inap(v, h)) / self.cm
        dndt = (self.ninf(v) - n) / self.taun(v)
        dhdt = (self.hinf(v) - h) / self.tauh(v)
        return dvdt, dndt, dhdt

    def h_in_v(self, v):
        n=self.ninf(v)
        # n=[]
        # for vs in v:
        #     n=np.append(self.ninf(vs))
        h=(self.i - self.il(v) - self.ina(v, n) - self.ik(v, n)  )/(self.gnap * self.mninf(v) * (v-self.ena))
        return h
    
    # def v_in_h(self,h):
    #     v=12*np.log((1/h)-1)-48
    #     return v

    def hnull(self,v):
        h=1/(1+np.exp((v+48.0)/12.0))
        return h
    
    def Main(self):
        """
        Main demo for the Butera Smith neuron model
        """
        # if not self.nullcline:
        X = odeint(self.dALLdt, [-55.0, 0.0, 0.6], self.t, args=(self,))
        V = X[:,0]
        n = X[:,1]
        h = X[:,2]
        ina = self.ina(V, n)
        ik = self.ik(V, n)
        il = self.il(V)
        inap=self.inap(V,h)

        
        if not self.phase:
            fig, axs = plt.subplots(3)
            fig.suptitle('butera and smith model reduced')
            fig.subplots_adjust(hspace=.5)
            axs[0].plot(self.t, V, 'k')
            axs[0].set(ylabel='V (mV)',xlabel="Time (ms)")
            axs[1].plot(self.t, ina, 'c', label='$I_{Na}$')
            axs[1].plot(self.t, ik, 'y', label='$I_{K}$')
            axs[1].plot(self.t, il, 'm', label='$I_{L}$')
            axs[1].plot(self.t, inap, 'b', label='$I_{NaP}$')

            axs[1].set(ylabel='Currents')
            axs[1].legend()
            axs[2].plot(self.t, n, 'r', label='n')
            axs[2].plot(self.t, h, 'g', label='h')
            axs[2].set(ylabel='Gating Variables')
            axs[2].legend()
            # i_inj_values = [self.I_inj(t) for t in self.t]
            # axs[3].plot(self.t, i_inj_values, 'k')
            # axs[3].set(xlabel='t (ms)',ylabel='$I_{inj}$ ($\\mu{A}/cm^2$)')

            # plt.savefig('KK-I10.png', format = 'png')
            plt.show()

        # if self.phase:
        #     fig, axs = plt.subplots(2)
        #     fig.suptitle('Butera reduced nullclines')
        #     fig.subplots_adjust(hspace=.5)
        #     axs[0].plot(V, h, 'k')
        #     # axs[1].set(ylabel='h',xlabel="V (mV)")
        #     # axs[0].plot(self.t, V, 'k')
        #     # axs[0].set(ylabel='V (mV)',xlabel="Time (ms)")
        #     plt.show()

        if self.nullcline:
            v=np.arange(-65,-54,0.1)
            # fig, axs = plt.subplots(2)
            # fig.suptitle('Butera reduced nullclines')
            # fig.subplots_adjust(hspace=.5)
            # axs[0].plot(V, h, 'k')
            # h=np.arange(-10,10,0.1)
            y1=[]
            y2=[]
            for vs in np.nditer(v):
                y1.append(self.h_in_v(vs))
                y2.append(self.hnull(vs))
            # fig, axs = plt.subplots(2)
            # fig.suptitle('Butera reduced nullclines')
            # fig.subplots_adjust(hspace=.5)
            # axs[0].plot(v, y1, 'k')
            # axs[0].plot(v, y2, 'k')
            # axs[0].plot(V, h, 'r')
            # axs[0].set(ylabel='h',xlabel="V (mV)")
            # axs[0].set(ylabel='h',xlabel="V (mV)")
            # axs[1].plot(self.t, self.t, 'k')
            # axs[1].set(ylabel='V (mV)',xlabel="Time (ms)")
            plt.title('Butera reduced nullclines')
            plt.plot(v, y1, 'g',linewidth=2)
            plt.plot(v, y2, 'b',linewidth=2)
            plt.plot(V, h, 'r',linewidth=2)
            plt.xlabel("V (mV)")
            plt.ylabel('h')
            plt.show()


if __name__ == '__main__':
    runner=Butera(50000.0,-65.0,reduction=True,phase=True,nullcline=True)
    runner.Main()

