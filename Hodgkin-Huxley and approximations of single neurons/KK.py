import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class HodgkinHuxley():
    """Full Hodgkin-Huxley Model implemented in Python"""
    def __init__(self,I_app):
        self.I_app=I_app

    C_m  =   1.0
    """membrane capacitance, in uF/cm^2"""

    g_Na = 120.0
    """Sodium (Na) maximum conductance, in mS/cm^2"""

    g_K  =  36.0
    """Postassium (K) maximum conductance, in mS/cm^2"""

    g_L  =   0.3
    """Leak maximum conductance, in mS/cm^2"""

    E_Na =  50.0
    """Sodium (Na) Nernst reversal potential, in mV"""

    E_K  = -77.0
    """Postassium (K) Nernst reversal potential, in mV"""

    E_L  = -54.387
    """Leak Nernst reversal potential, in mV"""

    t = np.arange(0.0, 700.0, 0.01)
    """ The time to integrate over """

    # I_app = 10
    # """ Applied current """
    
    def alpha_m(self, V):
        """Na channel activation gate opening rate. Function of membrane voltage"""
        return 0.1*(V+40.0)/(1.0 - np.exp(-(V+40.0) / 10.0))

    def beta_m(self, V):
        """Na channel activation gate closing rate. Function of membrane voltage"""
        return 4.0*np.exp(-(V+65.0) / 18.0)

    def alpha_h(self, V):
        """Na channel inactivation gate opening rate. Function of membrane voltage"""
        return 0.07*np.exp(-(V+65.0) / 20.0)

    def beta_h(self, V):
        """Na channel inactivation gate closing rate. Function of membrane voltage"""
        return 1.0/(1.0 + np.exp(-(V+35.0) / 10.0))

    def alpha_n(self, V):
        """K channel activation gate opening rate. Function of membrane voltage"""
        return 0.01*(V+55.0)/(1.0 - np.exp(-(V+55.0) / 10.0))

    def beta_n(self, V):
        """K channel activation gate closing rate. Function of membrane voltage"""
        return 0.125*np.exp(-(V+65) / 80.0)

    def I_Na(self, V, m, h):
        """
        Membrane current (in uA/cm^2)
        Sodium 
        |  :param V:
        |  :param m:
        |  :param h:
        |  :return:
        """
        return self.g_Na * m**3 * h * (V - self.E_Na)

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

    def I_inj(self, t):
        """
        External Current
        |  :param t: time
        |  :return: step up to 10 uA/cm^2 at t>100
        |           step down to 0 uA/cm^2 at t>600
        """
        return self.I_app*(t>100) - self.I_app*(t>600)

    @staticmethod
    def dALLdt(X, t, self):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
        """
        V, m, h, n = X

        dVdt = (self.I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        dmdt = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m
        dhdt = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h
        dndt = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n
        return dVdt, dmdt, dhdt, dndt

    def Main(self):
        """
        Main demo for the Hodgkin Huxley neuron model
        """

        X = odeint(self.dALLdt, [-65, 0.05, 0.6, 0.32], self.t, args=(self,))
        V = X[:,0]
        m = X[:,1]
        h = X[:,2]
        n = X[:,3]
        ina = self.I_Na(V, m, h)
        ik = self.I_K(V, n)
        il = self.I_L(V)

        threshold = 0
        threshold_reached = False

        num_of_peaks = 0
        prev = -9999999999
        for v in np.nditer(V):
            if prev < v and v > threshold and not threshold_reached:
                num_of_peaks += 1
                threshold_reached = True
            if prev > v:
                threshold_reached = False
            prev = v
        return num_of_peaks

    #     fig, axs = plt.subplots(4)
    #     fig.suptitle('Hodgkin-Huxley')
    #     fig.subplots_adjust(hspace=.5)
    #     axs[0].plot(self.t, V, 'k')
    #     axs[0].set(ylabel='V (mV)')
    #     axs[1].plot(self.t, ina, 'c', label='$I_{Na}$')
    #     axs[1].plot(self.t, ik, 'y', label='$I_{K}$')
    #     axs[1].plot(self.t, il, 'm', label='$I_{L}$')
    #     axs[1].set(ylabel='Currents')
    #     axs[1].legend()
    #     axs[2].plot(self.t, m, 'r', label='m')
    #     axs[2].plot(self.t, h, 'g', label='h')
    #     axs[2].plot(self.t, n, 'b', label='n')
    #     axs[2].set(ylabel='Gating Variables')
    #     axs[2].legend()
    #     i_inj_values = [self.I_inj(t) for t in self.t]
    #     axs[3].plot(self.t, i_inj_values, 'k')
    #     axs[3].set(xlabel='t (ms)',ylabel='$I_{inj}$ ($\\mu{A}/cm^2$)')

    #     plt.savefig('HH-I10.png', format = 'png')

    #     plt.show()


#KK model    
    # def dALLdt(X, t, self):
    #     """
    #     Integrate

    #     |  :param X:
    #     |  :param t:
    #     |  :return: calculate membrane potential & activation variables
         
    #      i think if i replace h=1-n 
    #      and m=m_infinity= alpha over alpha plus beta should work
    #     """
    #     V, n = X
    #     m_infinity=self.alpha_m(V)/(self.alpha_m(V)+self.beta_m(V))
    #     dVdt = (self.I_inj(t) - self.I_Na(V, m_infinity, 1-n) - self.I_K(V, n) - self.I_L(V)) / self.C_m
    #     #dmdt = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m
    #     #dhdt = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h
    #     dndt = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n
    #     return dVdt, dndt

  
    # def Main(self):
    #     """
    #     Main demo for the Hodgkin Huxley neuron model
    #     """

    #     X = odeint(self.dALLdt, [-65, 0.32], self.t, args=(self,))
    #     V = X[:,0]
    #     n = X[:,1]
    #     m_infinity=self.alpha_m(V)/(self.alpha_m(V)+self.beta_m(V))
    #     ina = self.I_Na(V,m_infinity , 1-n)
    #     ik = self.I_K(V, n)
    #     il = self.I_L(V)

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

    

    #     fig, axs = plt.subplots(4)
    #     fig.suptitle('Krinsky-Kokoz reduction')
    #     fig.subplots_adjust(hspace=.5)
    #     axs[0].plot(self.t, V, 'k')
    #     axs[0].set(ylabel='V (mV)')
    #     axs[1].plot(self.t, ina, 'c', label='$I_{Na}$')
    #     axs[1].plot(self.t, ik, 'y', label='$I_{K}$')
    #     axs[1].plot(self.t, il, 'm', label='$I_{L}$')
    #     axs[1].set(ylabel='Currents')
    #     axs[1].legend()
    #     axs[2].plot(self.t, m_infinity, 'r', label='m_infinity')
    #     axs[2].plot(self.t, 1-n, 'g', label='h=1-n')
    #     axs[2].plot(self.t, n, 'b', label='n')
    #     axs[2].set(ylabel='Gating Variables')
    #     axs[2].legend()
    #     i_inj_values = [self.I_inj(t) for t in self.t]
    #     axs[3].plot(self.t, i_inj_values, 'k')
    #     axs[3].set(xlabel='t (ms)',ylabel='$I_{inj}$ ($\\mu{A}/cm^2$)')

    #     plt.savefig('KK-I10.png', format = 'png')

    #     plt.show()


    #     #F-I curve
    # def Main(self):
    #     """
    #     Main demo for the Hodgkin Huxley neuron model
    #     """

    #     X = odeint(self.dALLdt, [-65, 0.32], self.t, args=(self,))
    #     V = X[:,0]
    #     n = X[:,1]
    #     m_infinity=self.alpha_m(V)/(self.alpha_m(V)+self.beta_m(V))
    #     ina = self.I_Na(V,m_infinity , 1-n)
    #     ik = self.I_K(V, n)
    #     il = self.I_L(V)

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

        # fig, axs = plt.subplots(4)
        # fig.suptitle('Krinsky-Kokoz reduction')
        # fig.subplots_adjust(hspace=.5)
        # axs[0].plot(self.t, V, 'k')
        # axs[0].set(ylabel='V (mV)')
        # axs[1].plot(self.t, ina, 'c', label='$I_{Na}$')
        # axs[1].plot(self.t, ik, 'y', label='$I_{K}$')
        # axs[1].plot(self.t, il, 'm', label='$I_{L}$')
        # axs[1].set(ylabel='Currents')
        # axs[1].legend()
        # axs[2].plot(self.t, m_infinity, 'r', label='m_infinity')
        # axs[2].plot(self.t, 1-n, 'g', label='h=1-n')
        # axs[2].plot(self.t, n, 'b', label='n')
        # axs[2].set(ylabel='Gating Variables')
        # axs[2].legend()
        # i_inj_values = [self.I_inj(t) for t in self.t]
        # axs[3].plot(self.t, i_inj_values, 'k')
        # axs[3].set(xlabel='t (ms)',ylabel='$I_{inj}$ ($\\mu{A}/cm^2$)')

        # plt.savefig('KK-I10.png', format = 'png')

        # plt.show()

def plotFI():
    t=.500
    f=[]
    I_apps=list(range(0,51))
    freqKK=[0.0, 0.0, 50.0, 58.0, 62.0, 66.0, 68.0, 72.0, 74.0, 76.0, 78.0, 80.0, 82.0, 84.0, 86.0, 88.0, 90.0, 90.0, 92.0, 94.0, 94.0, 96.0, 98.0, 98.0, 100.0, 100.0, 102.0, 102.0, 104.0, 106.0, 106.0, 108.0, 108.0, 108.0, 110.0, 110.0, 112.0, 112.0, 114.0, 114.0, 116.0, 116.0, 116.0, 118.0, 118.0, 118.0, 120.0, 120.0, 122.0, 122.0, 122.0]
    freqHH=[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 4.0, 60.0, 64.0, 66.0, 70.0, 72.0, 74.0, 76.0, 78.0, 80.0, 82.0, 82.0, 84.0, 86.0, 88.0, 88.0, 90.0, 92.0, 92.0, 94.0, 96.0, 96.0, 98.0, 98.0, 100.0, 100.0, 102.0, 102.0, 104.0, 104.0, 106.0, 106.0, 108.0, 108.0, 110.0, 110.0, 112.0, 112.0, 112.0, 114.0, 114.0, 116.0, 116.0, 116.0, 118.0]
    # for I_app in I_apps:
    #     hh=HodgkinHuxley(I_app=I_app)
    #     n=hh.Main()
    #     frequency=n/t
    #     f.append(frequency)
    #     print("n hh",n)
    #     print("f hh",f)
    plt.plot(I_apps,freqHH)
    plt.title("f-I curve HH model")
    plt.xlabel("$I_{inj}$ ($\\mu{A}/cm^2$)")
    plt.ylabel("frequency (mHz)")
    plt.savefig('HH-fI.png', format = 'png')
    plt.show()

        

if __name__ == '__main__':
    plotFI()
