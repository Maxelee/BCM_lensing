import illustris_python as il
import numpy as np
import pandas as pd
from nbodykit.cosmology import Planck15
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import minimize
from BCM_lensing.utils import *

class BCM_COMPONENT:
    """
    Hold all of the parameters an functions necessary to the BCM. This in cludes key halo information
    as well as the BCM parameter values, and values for computing the fractions. The values are mimicked
    from Arico 2020. BCM is a parent class that all components inherit from
    """
    def __init__(self, r_200, m_200, c, rho_s, 
                 M1=8.63e1, M_c=3.3e3, beta=0.12, eta=0.54,
                 omega_b=0.0455, omega_m=0.272, 
                 alpha = -1.779, delta = 4.394, gamma = 0.547, epsilon=0.023):

        # Halo Parameters
        self.r_200   = r_200
        self.m_200   = m_200
        self.c       = c
        self.rho_s   = rho_s

        # BCM Parameters
        self.M1      = M1
        self.M_c     = M_c
        self.omega_b = omega_b
        self.omega_m = omega_m
        self.beta    = beta
        self.eta     = eta

        # F parameters
        self.alpha = alpha
        self.delta = delta
        self.gamma = gamma
        self.epsilon =  epsilon
        self.f_CG = self.build_f_CG()
        self.f_BG = self.build_f_BG()
        self.f_EG = self.build_f_EG()
        self.f_RDM = self.build_f_RDM()

    def clip(self, rho, r):
        try:
            rho[r>self.r_200] = 0
        except:
            if r>self.r_200:
                rho=0
        return rho

    def M(self, rho_func, int_params, r):
        """
        Compute mass by integrating. This method is much slower than using analytic masses
        but can be used to check accuracy of Mass calculations. Also it is used for the 
        Bound gas profile which has no analytic form.
        """
        if (type(r) ==float) or (type(r) ==np.float32) or (type(r) ==np.float64):
            return integrate_over_r(r, (rho_func, int_params))
        else:
            return integrate_smart(r, (rho_func, int_params))

    def build_f_CG(self, M=None):
        """
        Fraction of the mass in Stellar component (central galaxy component) baryons
        """
        if not M:
            M = self.m_200
        return self.epsilon * (self.M1/M) * 10**(self.g(np.log10(M/self.M1)) - self.g(0))

    def build_f_BG(self, M=None):
        """
        fraction of mass in Hot bound gas baryons
        """
        if not M:
            M = self.m_200
        return self.omega_b/self.omega_m /(1+(self.M_c/M)**self.beta)

    def build_f_EG(self, M=None):
        """
        Fraction of gas in Ejected Gas baryons (computed by subtracting all other fractions)
        """
        if not M:
            M = self.m_200
        return self.omega_b/self.omega_m - self.f_CG - self.f_BG

    def build_f_RDM(self):
        """
        Fraction of gas in Relaxed Dark Matter
        """
        return (1-self.omega_b/self.omega_m)

    def g(self, x):
        """
        Function needed for computing the central galaxy and the ejected gas components
        """

        res =  -np.log10(10**(self.alpha * x)+1) +self.delta  *(np.log10(1 + np.exp(x))**self.gamma)/(1 + np.exp(10**-x))
        return res

class CG(BCM_COMPONENT):
    """
    Central Galaxy / Stellar component of the Baryonic correction model. Inherets from BCM_COMPONENT class, 
    And returns analytic density and mass calculations. 
    """
    def __init__(self, r_200, m_200, c, rho_s, R_h_mult=0.015,
                 M1=8.6e1, M_c=3.3e3, beta=.12, eta=.54,
                 omega_b=0.0486, omega_m=0.3089037):

        BCM_COMPONENT.__init__(self, r_200, m_200, c, rho_s,
                 M1=M1,  M_c=M_c, beta=beta, eta=eta,
                 omega_b=omega_b, omega_m=omega_m)

        self.R_h = R_h_mult * self.r_200



        self.int_params = []

    def density(self, r, M=None, clipped=True):
        if not M:
            M = self.m_200

        rho=  M/(4 * np.pi**(3/2) * self.R_h * r**2) * np.exp(-(r/(2*self.R_h))**2)
        if clipped:
            return self.clip(rho, r)*  self.f_CG

        return rho *  self.f_CG

    def Mass(self, r):
        return self.f_CG * self.m_200 * erf(r/(2*self.R_h))

class BG(BCM_COMPONENT):
    """
    Bound gas component of the Baryonic correction model. Inherets from BCM_COMPONENT class, 
    And returns analytic density and mass calculations. 
    """
    def __init__(self, r_200, m_200, c, rho_s,
                 M1=8.6e1, M_c=3.3e3, beta=.12, eta=.54,
                 omega_b=0.0486, omega_m=0.3089037):

        BCM_COMPONENT.__init__(self, r_200, m_200, c, rho_s,
                 M1=M1,  M_c=M_c, beta=beta, eta=eta,
                 omega_b=omega_b, omega_m=omega_m)

        self.gamma_c = self._gamma()
        self.int_params = []

        self._normalize()
        self._build_Mass()

    def _gamma(self):
        """
        polytropic index function which depends on the concentration parameter of the halo
        """
        num = (1+3 * self.c /np.sqrt(5)) * np.log(1 + self.c/np.sqrt(5))
        denom = (1+ self.c/np.sqrt(5)) * np.log(1+self.c/np.sqrt(5)) - self.c/np.sqrt(5)
        return num/denom

    def _build_Mass(self):
        """
        Because no analytic form of the mass function exists, I create an interpolation
        function from integration. Calling Mass then calls this interpolation function
        """
        rs = np.logspace(np.log10(10), np.log10(self.r_200*2), 20)
        masses = np.array(self.M(self.density, self.int_params, rs))
        self.Mass_func = interp1d(rs, masses, fill_value='extrapolate')

    def _normalize(self, M=None):
        """
        The normalization for the density function is somewhat complicated... This is my implementation
        though I would like to find a way to test this
        """
        #### TODO: Develop a test to check the accuracy/validity of my implementation 

        if not M:
            M = self.m_200

        r_s = self.r_200/self.c
        x_b = self.r_200/r_s/np.sqrt(5) - .040
        x_a = self.r_200/r_s/np.sqrt(5) + .040
        below = (x_b**-1 * np.log(1+x_b))**self.gamma_c
        above =  x_a**-1 *(1+x_a)**-2

        ratio = below/above

        above_integral = lambda r, r_s=r_s: (r/r_s)**-1 *(1+r/r_s)**-2 * 4 * np.pi *r **2
        below_integral = lambda r, c=self.c, r_s=r_s: ((r/r_s)**-1 * np.log(1+(r/r_s)))**self.gamma_c * 4 * np.pi * r**2

        a = quad(above_integral, self.r_200/np.sqrt(5), self.r_200, args=(r_s))[0]
        b = quad(below_integral, 0, self.r_200/np.sqrt(5), args=(self.c, r_s))[0]



        self.y0 = M / (ratio * a + b)
        self.y1 = self.y0 * ratio

    def Mass(self, r):
        """
        Call the interpolation function at a given radius
        """
        return self.Mass_func(r)

    def density(self, r, M=None, clipped=False):
        """
        Hydrostatic equilibrium upto r_200/sqrt(5) and NFW slope after.
        """
        if not M:
            M = self.m_200

        r_s = self.r_200/self.c
        x = r/r_s

        below = (x**-1 * np.log(1+x))**self.gamma_c
        above =  x**-1 *(1+x)**-2

        try:
            above[r <=self.r_200/np.sqrt(5)] = 0
            above[r >self.r_200] = 0
            below[r > self.r_200/np.sqrt(5)]= 0

        except:
            if (r<self.r_200/np.sqrt(5)) or (r>self.r_200):
                above = 0
            if r > self.r_200/np.sqrt(5):
                below=0

        res = self.y1*above+self.y0*below
        rho = np.array(res * self.f_BG)

        if clipped:
            return self.clip(rho, r)

        return rho 

class EG(BCM_COMPONENT):
    """
    Ejected gas component of the Baryonic correction model. Inherets from BCM_COMPONENT class, 
    And returns analytic density and mass calculations. 
    """


    def __init__(self, r_200, m_200, c, rho_s,
                 M1=8.6e1, M_c=3.3e3, beta=.12, eta=.54,
                 omega_b=0.0486, omega_m=0.3089037):

        BCM_COMPONENT.__init__(self, r_200, m_200, c, rho_s,
                 M1=M1,  M_c=M_c, beta=beta, eta=eta,
                 omega_b=omega_b, omega_m=omega_m)

        self.r_esc = 1/2 * np.sqrt(200) * r_200
        self.r_ej = self.eta * .75 * self.r_esc

    def density(self, r, M=None):
        if not M:
            M=self.m_200
        return self.f_EG * M/(2 * np.pi * self.r_ej**2)**(3/2) * np.exp(-1/2 * (r/self.r_ej)**2)

    def Mass(self, r):
        factor = 4 * np.pi * self.m_200 / (2*np.pi*self.r_ej**2)**(3/2)*self.f_EG
        return factor * self.r_ej**2 * (np.sqrt(np.pi/2) * self.r_ej * erf(np.sqrt(2)/2 * r/self.r_ej) - r * np.exp(-1/2 * (r/self.r_ej)**2))

class RDM(BCM_COMPONENT):
    """
    relaxed DM component of the Baryonic correction model. Inherets from BCM_COMPONENT class, 
    And returns analytic density and mass calculations. 

    RDM has routines for fitting the RDM parameter xi, computing the mass, and taking the derivative
    of the mass equation using finite differencing to compute the density profile. 
    """
    def __init__(self,cg, bg, eg, a=0.3, n=2, tol=1e-2,
                 M1=8.6e1, M_c=3.3e3, beta=.12, eta=.54,
                 omega_b=0.0486, omega_m=0.3089037):

        BCM_COMPONENT.__init__(self, cg.r_200, cg.m_200, cg.c, cg.rho_s,
                 M1=M1,  M_c=M_c, beta=beta, eta=eta,
                 omega_b=omega_b, omega_m=omega_m)


        # xi Minimization Parameters
        self.a       = a
        self.n       = n
        self.tol     = tol
        self.xis     = []
        self.diffs    = []

        # Initialize the BCM components
        self.cg = cg
        self.eg = eg
        self.bg = bg

    def _run_one_r(self, r, m, xi_i):
        xi = self.xi(m, self.Mf(r, m))
        diff = np.abs(xi_i - xi)
        return  xi, diff

    def _check_convergence(self, diff):
        self.diffs.append(diff)
        if diff <self.tol:
            return True
        else:
            return False

    def _Mass_fd(self, ri, masses, xi, delta=1e-3):
        """
        Central finite differencing for computing density function
        """
        above = self.Mass(ri + delta, masses, xi)
        below = self.Mass(ri - delta, masses, xi)
        return (above - below)/(2*delta)

    def xi(self, Mi, Mf):
        """
        Analytical caluclation for xi
        """
        val =  1  + self.a * (Mi/Mf)**self.n - self.a
        return val

    def Mf(self, r, M_DM):

        M_CG = self.cg.Mass(r)

        M_BG = self.bg.Mass(r)

        M_EG = self.eg.Mass(r)

        return   M_CG + M_BG + M_EG + self.f_RDM * M_DM

    def run_xi(self, ri, masses):
        """
        xi calculation function. Iteratively solve the xi function and update
        final masses to compute the value of xi when it converges. Perform this
        iterative procedure for every r value corresponding to a spherical shell.
        Normalize the xi values such that the xi at r_200 = 1. See arico appendix
        A or Schneider 15 for more details.
        """
        xi = 1
        for rs, m in zip(ri, masses):
            converged = False
            xi_i = 1
            while not converged:
                r = rs / xi
                xi, diff = self._run_one_r(r, m, xi_i)
                xi_i = xi
                converged = self._check_convergence(diff)
            self.xis.append(xi_i)
        self.xi_func = interp1d(ri, np.array(self.xis), fill_value='extrapolate')


        return np.array(self.xis)

    def Mass(self, ri, masses, xi):
        """
        For Mass computation, build an interpolation function from
        Dark matter only masses to radial shells. Then plug in the radii from RDM
        corresponding to ri*xi to find the masses of relaxed dark matter halo shells.
        """
        return self.f_RDM* self.M_DMO_interp(ri/xi)

    def build_MassFunc(self, ri, masses):
        self.M_DMO_interp = interp1d(ri, masses, fill_value='extrapolate', bounds_error=False)

    def build_MassFunc(self, ri, masses):
        self.M_DMO_interp = interp1d(ri, masses, fill_value='extrapolate', bounds_error=False)

    def density(self, ri, masses, xi, delta=1e-3, clipped=True):
        """
        Compute the derivative of the mass function and normalize by a volume term to findt
        the density profile of relaxed dark matter
        """
        self.build_MassFunc(ri,masses)
        rho= 1/(4*np.pi*ri**2) *  self._Mass_fd(ri, masses, xi)
        if clipped:
            return self.clip(rho, ri)
        return rho
