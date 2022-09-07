import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class Butera():
    """Full Butera Model implemented in Python"""
    # def __init__(self,I_app):
    #     self.I_app=I_app

    def __init__(self,tmax,el):
        self.tmax=tmax
        self.C_m  =   21.0
        """membrane capacitance, in uF/cm^2"""

        self.g_Na = 28.0
        """Sodium (Na) maximum conductance, in mS/cm^2"""

        self.g_K  =  11.2
        """Postassium (K) maximum conductance, in mS/cm^2"""

        self.g_L  =   2.8
        """Leak maximum conductance, in mS/cm^2"""

        self.E_Na =  50.0
        """Sodium (Na) Nernst reversal potential, in mV"""

        self.G_NaP=2.8

        self.tau_bar=10000

        self.E_K  = -85.0
        """Postassium (K) Nernst reversal potential, in mV"""

        self.E_L  = el
        """Leak Nernst reversal potential, in mV,"""
        # the thing V_L we need to change for PS2

        self.t = np.arange(0.0, tmax, 0.01)
        """ The time to integrate over """

        self.I_app = 0.0
        """ Applied current """
    
    # def alpha_m(self, V):
    #     """Na channel activation gate opening rate. Function of membrane voltage"""
    #     return 0.1*(V+40.0)/(1.0 - np.exp(-(V+40.0) / 10.0))

    # def beta_m(self, V):
    #     """Na channel activation gate closing rate. Function of membrane voltage"""
    #     return 4.0*np.exp(-(V+65.0) / 18.0)

    # def alpha_h(self, V):
    #     """Na channel inactivation gate opening rate. Function of membrane voltage"""
    #     return 0.07*np.exp(-(V+65.0) / 20.0)

    # def beta_h(self, V):
    #     """Na channel inactivation gate closing rate. Function of membrane voltage"""
    #     return 1.0/(1.0 + np.exp(-(V+35.0) / 10.0))

    # def alpha_n(self, V):
    #     """K channel activation gate opening rate. Function of membrane voltage"""
    #     return 0.01*(V+55.0)/(1.0 - np.exp(-(V+55.0) / 10.0))

    # def beta_n(self, V):
    #     """K channel activation gate closing rate. Function of membrane voltage"""
    #     return 0.125*np.exp(-(V+65) / 80.0)

    def I_Na(self, V, n):
        """
        Membrane current (in uA/cm^2)
        Sodium 
        |  :param V:
        |  :param m:
        |  :param h:
        |  :return:
        """
        return self.g_Na * self.m_inf(V)**3 * (1-n) * (V - self.E_Na)

    def I_K(self, V, n):
        """
        Membrane current (in uA/cm^2)
        Potassium 
        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_K  * n**4 * (V - self.E_K)

    def I_L(self, V):
        """
        Membrane current (in uA/cm^2)
        Leak
        |  :param V:
        |  :return:
        """
        return self.g_L * (V - self.E_L)

    # def I_inj(self, t):
    #     """
    #     External Current
    #     |  :param t: time
    #     |  :return: step up to 10 uA/cm^2 at t>100
    #     |           step down to 0 uA/cm^2 at t>600
    #     """
    #     return self.I_app*(t>100) - self.I_app*(t>600)

    def X_inf(self,V,vt,sig):
        return 1/(1+np.exp((V-vt)/sig))

    def Tau_x(self,V,vt,sig,tau):
        return tau/(np.cosh((V-vt)/(2*sig)))

    def I_l(self,V):
        return self.g_L*(V-self.E_L)

    def m_inf(self, V):
        return self.X_inf(V,-34.0,-5.0)

    def n_inf(self,V):
        return self.X_inf(V,-29.0,-4.0)

    def tau_n(self,V):
        return self.Tau_x(V,-29.0,-4.0,10.0)

    def m_inf(self, V):
        return self.X_inf(V,-40.0,-6.0)

    def h_inf(self, V):
        return self.X_inf(V,-48.0,6.0)

    def tau_h(self, V):
        return self.Tau_x(V,-48.0,6.0,self.tau_bar)

    def I_NaP(self,V,h):
        return self.G_NaP*self.m_inf(V)*h*(V-self.E_Na)



    @staticmethod
# butera model    
    def dALLdt(X, t, self):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
        
        i think if i replace h=1-n 
        and m=m_infinity= alpha over alpha plus beta should work
        """
        V, n,h = X
        dVdt=(self.I_app-self.I_l(V)-self.I_Na(V,n)-self.I_K(V,n)-self.I_NaP(V,h))/self.C_m
        dndt=(self.n_inf(V)-n)/self.tau_n(V)
        dhdt=(self.h_inf(V)-h)/self.tau_h(V)

        # m_infinity=self.alpha_m(V)/(self.alpha_m(V)+self.beta_m(V))
        # dVdt = (self.I_inj(t) - self.I_Na(V, m_infinity, 1-n) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        #dmdt = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m
        #dhdt = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h
        # dndt = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n
        return dVdt, dndt,dhdt


    def Main(self):
        """
        Main demo for the Hodgkin Huxley neuron model
        """

        X = odeint(self.dALLdt, [-55.0, 0.0,0.6], self.t, args=(self,))
        V = X[:,0]
        n = X[:,1]
        h = X[:,2]
        # m_infinity=self.alpha_m(V)/(self.alpha_m(V)+self.beta_m(V))
        ina = self.I_Na(V,n)
        ik = self.I_K(V, n)
        il = self.I_l(V)
        inap=self.I_NaP(V,h)

    #     threshold = 0
    #     threshold_reached = False

    #     num_of_peaks = 0
    #     prev = -9999999999
    #     for v in np.nditer(V):
    #         if prev < v and v > threshold and not threshold_reached:
    #             num_of_peaks += 1
    #             threshold_reached = True
    #         if prev > v:
    #             threshold_reached = False
    #         prev = v
    #     return num_of_peaks

    

        fig, axs = plt.subplots(3)
        fig.suptitle('butera and smith model using NaP')
        fig.subplots_adjust(hspace=.5)
        axs[0].plot(self.t, V, 'k')
        axs[0].set(ylabel='V (mV)')
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


    # Butera model
    # butera and smith model using NaP
# par cm=21,I=0
# xinf(v,vt,sig)=1/(1+exp((v-vt)/sig))
# eq 3 
# done

# taux(v,vt,sig,tau)=tau/cosh((v-vt)/(2*sig))
# eq 4

# #leak
# il=gl*(v-el)
# par gl=2.8,el=-65
#done


# # fast sodium --  h=1-n
# minf(v)=xinf(v,-34,-5)
# ina=gna*minf(v)^3*(1-n)*(v-ena)
# par gna=28,ena=50
#done

# # delayed rectifier
# ninf(v)=xinf(v,-29,-4)
# taun(v)=taux(v,-29,-4,10)
# ik=gk*n^4*(v-ek)
# par gk=11.2,ek=-85
#done

# # NaP
# mninf(v)=xinf(v,-40,-6)
# hinf(v)=xinf(v,-48,6)
# tauh(v)=taux(v,-48,6,taubar)
# par gnap=2.8,taubar=10000
# inap=gnap*mninf(v)*h*(v-ena)


# v' = (i-il-ina-ik-inap)/cm
# n'=(ninf(v)-n)/taun(v)
# h'=(hinf(v)-h)/tauh(v)
# @ total=40000,dt=1,meth=cvode,maxstor=100000
# @ tol=1e-8,atol=1e-8
# @ xlo=0,xhi=40000,ylo=-80,yhi=20
# done


if __name__ == '__main__':
    runner=Butera(10000.0,-60.0)
    runner.Main()
