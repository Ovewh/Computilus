import numpy as np
DATADIR = '../data/'


# Functions for initializing psi and zeta
def sine(x):
    return np.sin(4*np.pi*x)


def sine_der(x):
    return - 16*np.pi**2 * np.sin(4*np.pi*x)


def gauss(x, x0, sigma):
    return np.exp(- ((x - x0)/sigma)**2)


def gauss_der(x, x0, sigma):
    return 2*(2*(x-x0)**2 - sigma**2) * gauss(x, x0, sigma) / sigma**4


def poisson1d_periodic(p, b, dx, nx, target=1e-6, iter=10000):
    count = 0
    diff = 1
    while (diff > target and count < iter):
        diff = 0
        pn = p.copy()
        # Borders
        # p[-1] = 0.5 * (pn[1] + pn[-2] - b[-1] * dx**2)
        # p[0] = 0.5 * (pn[1] + pn[-2] - b[0] * dx**2)

        for i in range(0, nx):
            p[i] = 0.5 * (pn[(i+1) % nx] + pn[(i-1) % nx] - b[i] * dx**2)
            diff += np.abs(pn[i] - p[i])
        count += 1
        diff /= nx
    print('Iterations: {}'.format(count))
    return p


def poisson1d_bounded(p, b, dx, nx, target=1e-6, iter=10000):
    count = 0
    diff = 1
    while (diff > target and count < iter):
        diff = 0
        pn = p.copy()
        for i in range(1, nx - 1):
            p[i] = 0.5 * (pn[i+1] + pn[i-1] - b[i] * dx**2)
            diff += np.abs(pn[i] - p[i])
        count += 1
        diff /= nx
    print('Iterations: {}'.format(count))
    return p


def periodic(dx, t, init, advance, dt=0.1, x0=0.5, sigma=0.1, save=True):
    nx = int(1/dx)
    nt = int(t/dt)
    alpha = dt/dx
    x = np.linspace(0, 1 - dx, nx)
    psi_file = DATADIR + 'psi_periodic_' + advance.__name__ + '_' + '{}'.format(dt) + '_'
    zeta_file = DATADIR + 'zeta_periodic_' + advance.__name__ + '_' + '{}'.format(dt) + '_'
    # Initalize psi and zeta
    if init == sine:
        psi = init(x)
        zeta = sine_der(x)
        psi_file += 'sine.csv'
        zeta_file += 'sine.csv'
    else:
        psi = init(x, x0, sigma)
        zeta = gauss_der(x, x0, sigma)
        psi_file += 'gauss_{:.2f}.csv'.format(sigma)
        zeta_file += 'gauss_{:.2f}.csv'.format(sigma)
    zeta_n = zeta.copy()
    zeta_nn = zeta_n.copy()
    # for i in range(0, nx):
    #     zeta_n[i] = zeta[i] + dt/(2*dx) * (psi[(i+1) % nx] - psi[(i-1) % nx])
    # for i in range(0, nx):
    #     zeta_nn[i] = zeta_n[i] + dt/(2*dx) * (psi[(i+1) % nx] - psi[(i-1) % nx])
    print(psi_file, zeta_file)
    # TODO: Add function parameters to files
    if save:
        write(psi_file, psi, 0, mode='w')
        write(zeta_file, zeta, 0, mode='w')
    for n in range(nt):
        for i in range(0, nx):
            advance(zeta, zeta_nn, i, alpha, psi, nx)
        psi = poisson1d_periodic(psi, zeta, dx, nx)
        zeta_nn = zeta_n.copy()
        zeta_n = zeta.copy()
        if save:
            write(psi_file, psi, n*dt)
            write(zeta_file, zeta, n*dt)
    return psi, zeta


# Time stepping functions periodic BC
def centered(zeta, zeta_nn, i, alpha, psi, nx):
    zeta[i] = zeta_nn[i] - alpha * (psi[(i+1) % nx] - psi[(i-1) % nx])


def forward(zeta, zeta_nn, i, alpha, psi, nx):
    zeta[i] = zeta[i] - 0.5 * alpha * (psi[(i+1) % nx] - psi[(i-1) % nx])


def bounded(dx, t, init, advance, dt=0.1, x0=0.5, sigma=0.1, save=True):
    nx = int(1/dx + 1)
    nt = int(t/dt)
    alpha = dt/dx
    x = np.linspace(0, 1, nx)
    psi_file = DATADIR + 'psi_bounded_' + advance.__name__ + '_' + '{}'.format(dt) + '_'
    zeta_file = DATADIR + 'zeta_bounded_' + advance.__name__ + '_' + '{}'.format(dt) + '_'
    # Initalize psi and zeta
    if init == sine:
        psi = init(x)
        zeta = sine_der(x)
        psi_file += 'sine.csv'
        zeta_file += 'sine.csv'
    else:
        psi = init(x, x0, sigma)
        zeta = gauss_der(x, x0, sigma)
        psi_file += 'gauss_{:.2f}.csv'.format(sigma)
        zeta_file += 'gauss_{:.2f}.csv'.format(sigma)
    zeta_n = zeta.copy()
    zeta_nn = zeta_n.copy()
    print(psi_file, zeta_file)
    # TODO: Add function parameters to output files
    if save:
        write(psi_file, psi, 0, mode='w')
        write(zeta_file, zeta, 0, mode='w')
    for n in range(nt):
        for i in range(1, nx - 1):
            zeta[i] = zeta_nn[i] - alpha * (psi[i+1] - psi[i-1])
        psi = poisson1d_bounded(psi, zeta, dx, nx)
        zeta_nn = zeta_n.copy()
        zeta_n = zeta.copy()
        if save:
            write(psi_file, psi, n*dt)
            write(zeta_file, zeta, n*dt)
    print(psi[0], psi[-1])
    return psi, zeta


def write(filename, vector, t, mode='a'):
    with open(filename, mode) as file:
        file.write('{},'.format(t))
        for p in vector[:-1]:
            file.write('{},'.format(p))
        file.write(str(vector[-1]))
        file.write('\n')


if __name__ == '__main__':
    t = 150
    dx = 1/40
    sigma = 0.1
    x0 = 0.5

    # Bounded sine and gaussian
    bounded(dx, t, init=sine, advance=centered)
    bounded(dx, t, init=gauss, advance=centered, sigma=sigma, x0=x0)

    # Gaussian periodic
    for sigma in [0.08, 0.09, 0.10, 0.11, 0.12]:
        periodic(dx, t, init=gauss, advance=centered, sigma=sigma, x0=x0)

    # Sine periodic
    t = 1500
    periodic(dx, t, init=sine, advance=forward, dt=1)
    periodic(dx, t, init=sine, advance=centered, dt=1)
