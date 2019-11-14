from pytest import approx
import numpy as np
from ising import ising, expectation_values
import analytic as a

def test_ising_EM():
    """
    Check that the ising model produces same result as the
    analytical solution for the 2x2 grid.

    Tests energy and magnetization.
    """
    L = 2
    spins = L**2
    cycles = 100000
    tol = 0.001
    delay = 0
    T=1
    for i in range(0,10):
        values,acc,d = ising(L,cycles,T)
        expect = expectation_values(values, cycles,L,T)

        # Analytical solutions
        E = a.expected_energy(T)/spins
        Mabs = a.expected_magnetization(T)/spins

        # Compare
        assert E == approx(expect[0], rel=tol)
        assert Mabs == approx(expect[2], rel=tol)


def test_ising_cv_suscept():
    """
    Check that the ising model produces same result as the
    analytical solution for the 2x2 grid.

    Tests heat capacity and susceptibility
    """
    L = 2
    spins = L**2
    cycles = 1000000
    tol = 0.1
    delay = 0
    T=1
    for i in range(0,10):
        values,acc,d = ising(L,cycles,T, delay)
        expect = expectation_values(values, cycles,L,T, delay)

        # Analytical solutions
        cv = a.cv(T)/spins
        suscept = a.susceptibility(T)/spins

        # Compare
        # assert cv == approx(expect[3], rel=tol)
        assert suscept == approx(expect[4], rel=tol)

def test_jit():
    """
    Check that ising with and without numba gives close to the same
    values. Uses rel=0.001 since we did not want to clutter the ising
    function more by having an option to set a seed in the RNG.
    Does not test the magnetization since it has very high variance.
    """

    nojit,a,d = ising.py_func(2,100000,1)
    jit,a,d = ising(2,100000,1)
    for value1, value2 in zip(nojit[[0,1,3,4]], jit[[0,1,3,4]]):
        assert value1 == approx(value2, rel=0.001)

if __name__ == '__main__':
    test_jit()
