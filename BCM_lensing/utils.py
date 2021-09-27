import numpy as np
from scipy.integrate import quad

def make_r(xyz, xyz_COM):
    return np.sqrt(np.sum((xyz-xyz_COM)**2, axis=1))


def density(r, M):
    rho= M / (4/3 * np.pi * r**3)
    return rho

def integrate_over_r(r, f_params, r0=0):
    return quad(rho_integration,r0, r, f_params, epsabs=1.49e-02,epsrel=1.49e-02)[0]

def rho_integration(r, f, params):
    return 4*np.pi * r**2 * f(r, *params)

def integrate_smart(rs, f_params):
    M = 0
    Ms = []
    M = integrate_over_r(rs[0], f_params, 0)
    Ms.append(M)
    for i,r in enumerate(rs[1:]):
        M += integrate_over_r(r, f_params, rs[i])
        Ms.append(M)
    return Ms

def get_theta(z, r):
    return np.arccos(z/r)

def get_phi(y, x):
    return np.arctan2(y, x)

def get_r(x,y, z):
    return np.sqrt(x**2 + y**2 + z**2)

def get_spherical(point, center):
    res = point-center
    x, y, z = res[:, 0], res[:, 1], res[:, 2]
    r = get_r(x, y, z)
    theta = get_theta(z, r)
    phi = get_phi(y,x)
    return np.array([r, theta, phi])

def make_x(r, theta, phi):
    return r*np.cos(phi) * np.sin(theta)

def make_y(r, theta, phi):
    return r*np.sin(theta)*np.sin(phi)

def make_z(r, theta):
    return r*np.cos(theta)

def make_cartesian(spherical, center):
    r = spherical[0,:]
    theta = spherical[1, :]
    phi = spherical[2, :]
    x = make_x(r, theta, phi)
    y = make_y(r, theta, phi)
    z = make_z(r, theta)
    return np.array([x, y, z]).T + center


def chi2(model, obs, sig):
    return np.sum((model-obs)**2/sig**2)
