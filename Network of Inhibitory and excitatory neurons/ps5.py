import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class EINet():
    """Excitatory inhibitory network"""

    Jee = 2.9  # J1
    Jei = 1.0  # J2
    Jie = 8.0  # J3
    Jii = 1.0  # J4

    Ie = 0.8  # I1
    Ii = 1.3  # I2
    delta_Ii = 0.1  # I3

    t = np.arange(0, 100, 0.1)
    time_to_injection = 20
    # print('t', t)

    re_star = (Ie * (1 + Jii) - Ii * Jei) / ((1 + Jii) * (1 - Jee) + Jei * Jie)  # r1
    ri_star = (Ii * (1 - Jee) + Ie * Jie) / ((1 + Jii) * (1 - Jee) + Jei * Jie)  # r2

    def enull(self,re):
        nullcline1=(self.Ie - re + self.Jee *re)/self.Jei
        return nullcline1

    def inull(self,re,I):
        nullcline2=(I + self.Jie *re)/(1 + self.Jii)
        return nullcline2

    # print('re_star', re_star)
    # print('ri_star', ri_star)

    def get_Ii(self, t):
        if t < self.time_to_injection:
            return self.Ii
        else:
            return self.Ii + self.delta_Ii

    @staticmethod
    def phi(x):
        if x < 0:
            return 0
        elif 0 <= x <= 1:
            return x
        elif x > 1:
            return 1

    @staticmethod
    def dALLdt(X, t, self):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate firing rate of inhibitory neuron cluster and excitatory neuron cluster
        """
        re, ri = X

        Ii = self.get_Ii(t)
        # print('Ii', Ii, t)

        dredt = -re + self.phi(self.Jee * re - self.Jei * ri + self.Ie)
        dridt = -ri + self.phi(self.Jie * re - self.Jii * ri + Ii)

        return dredt, dridt

    def Main(self):
        """
        Main demo for the Hodgkin Huxley neuron model
        """
        X = odeint(self.dALLdt, [self.re_star, self.ri_star], self.t, args=(self,))
        re = X[:, 0]
        ri = X[:, 1]

        print('re', re)
        print('ri', ri)

        rest=np.arange(0.0,0.04,0.001)

        y1=[]
        y2=[]
        y3=[]
        for res in np.nditer(rest):
            y1.append(self.enull(res))
            y2.append(self.inull(res,self.Ii))
            y3.append(self.inull(res,self.Ii+self.delta_Ii))
            # y2.append(self.inull(res,1.3))
            # y3.append(self.inull(res,1.4))


        fig, axs = plt.subplots(3)
        fig.suptitle('EI-Net')
        fig.subplots_adjust(hspace=.5)

        axs0].plot(self.t, re, 'k')[
        axs[0].set(ylabel='re')

        axs[1].plot(self.t, ri, 'c')
        axs[1].set(ylabel='ri')

        axs[2].plot(re, ri, 'b')
        axs[2].plot(rest,y1 , 'r')
        axs[2].plot(rest, y2, 'g')
        axs[2].plot(rest, y3, 'g')
        axs[2].set(ylabel='ri', xlabel='re')

        

        # plt.title('Dynamics')
        # plt.plot(self.t, re, 'b',linewidth=2)
        # plt.plot(self.t, ri,  'r',linewidth=2)
        # plt.xlabel("t")
        # plt.ylabel('ri, re')


        
        
        
        # plt.title('Nullclines')
        # plt.plot(re, ri, 'b',linewidth=1)
        # plt.plot(rest, y1, 'r',linewidth=2)
        # plt.plot(rest, y2, 'g',linewidth=2)
        # plt.plot(rest, y3, 'g',linewidth=2)
        # plt.xlabel("re")
        # plt.ylabel('ri')




        plt.show()



if __name__ == '__main__':
    runner = EINet()
    runner.Main()
