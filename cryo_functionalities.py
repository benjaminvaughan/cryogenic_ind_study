import numpy as np
import matplotlib.pyplot as plt


class Cryo_Funs():
    def __init__(self):
        pass

    def conductive(self, A_rod, l_rod, Th, Tc, dt, nrods):
        # A_rod = contact area between supports and box
        # l_rod = length of the rod
        # Th = hot temperature
        # Tc = cold temperature
        # dt = time step
        # nrods = number of rods
        # print(A_rod, l_rod)
        return nrods * A_rod / l_rod * (k(Th)-k(Tc))* (Th-Tc) * dt

    def rad(self, e, r300, r77, Th, Tc, dt, A300):
        # e - emissivity
        # r300 - radius of the hot source
        # r77 - radius of the cold source
        # Th - hot temperature
        # Tc - cold temperature
        # A300 - area of the hot source
        return A300 * 1 / (1/e + r300/r77*(1 /e - 1)) *(Th**4 - Tc**4) * dt * 5.67e-8

    def heat_shield_rad(self, e, Th, Tc, dt, A300, n):
        #e - emissivity
        #Th - hot temp
        #Tc - cold temp
        #dt - time step
        #A300 - radiating area
        # n - number of sheets
        return (2 * (1/e + 1/e - 1) + (n - 1) * (2 - e) / e)**-1 * A300*(Th**4 - Tc**4) * dt * 5.67e-8

    def convective(self):
        return
