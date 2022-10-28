import numpy as np
import matplotlib.pyplot as plt

class Cryo_Funs():
    def __init__(self):
        pass

    def conductive(self, A_rod, l_rod, Th, Tc, dt, nrods, k):
        # A_rod = contact area between supports and box
        # l_rod = length of the rod
        # Th = hot temperature
        # Tc = cold temperature
        # dt = time step
        # nrods = number of rods
        # print(A_rod, l_rod)
        if Tc < 4.5: #the thermal conducivity interpolation can only go down to 4.5 K
            Tc = 4.5
        return nrods * A_rod / l_rod * (k(Th)-k(Tc))* (Th-Tc) * dt

    def rad(self, eh, ec, r300, r77, Th, Tc, dt, A300):
        # e - emissivity
        # r300 - radius of the hot source
        # r77 - radius of the cold source
        # Th - hot temperature
        # Tc - cold temperature
        # A300 - area of the hot source
        return A300 * 1 / (1/eh + r300/r77*(1 /ec - 1)) *(Th**4 - Tc**4) * dt * 5.67e-8

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

    def specific_heat(dQ, m, specific_heat, T):
        #dQ is the change in heat
        # m is the mass of the object being heated
        # specific heat is an interpolated function of the specific heat capacity
        # T is the current temperature of the object

        c = specific_heat(T)
        dT = dQ / (m*c)

def reach_equilibrium_temp(T_heat_shield, r_heat_shield, A_heat_shield, r_stage, rho_stage, r_tm, e_stycast, A_stage, stage_height, e_al, thickness, T_stage, Tcc, A_rod, l_rod, A_test, height, mass, nhours, dt):
    # If I wanted to spend a lot of time on this I suppose I could define class objects for each of the components so I'm not
    # carrying around this huge list of parameters.
    # alas...
    '''
    heat-shield params
    T_heat_shield - temperature of the heat shield (assumed to be held at 77 K )
    r_heat_shield - radius of the heat shield
    e_stycast - emissivity of the stycast
    A_heat_shield
    4-kelvin stage
    rho_stage - density of the stage material
    A_stage - area of the stage base plate
    r_stage - radius of the stage
    stage_height - height of the 4 K stage
    e_al - emissivity of the al
    thickness - thickness of stage base plate
    T_stage - Temperature of the stage (starts at 77 and iteratively decreases)
    Tcc - Temperature of coolant (in this case LHe which is ~4.2 K )
    A_rod - area of the rods supporting the 4 K stage
    l_rod - length of the rod supporting the 4 K stage
    Aluminum test mass
    A_test - area of the test mass in contact with the 4 K stage
    height - height of the test mass in contact with the 4 K stage
    r_tm - radius of the test mass (assuming it is cylindrical here)
    mass - mass of the test material
    other
    dt - time step in seconds
    nhours - number of hours to run the simulation for
    '''
    cf = Cryo_Funs()
    T_tm = T_heat_shield #assuming that the test mass and the 4 K stage are initially in thermal equilibirum with the heat shield
    al_specific_heat = al_specific_heat_interpolator()
    al_therm_cond = al_interpolator()
    g10_therm_cond = g10_interpolator()
    V_stage = A_stage * stage_height
    for time in range(0, nhours*3600, dt): #3600 to conv hours to seconds, start at 0 seconds, and go by chunks of dt

        #energy into the 4k stage
        #77 K rad - > 4 K rad
        dq_rad_heat_shield = cf.rad(e_stycast, e_al, r_heat_shield, r_stage, T_heat_shield, T_stage, dt, A_heat_shield)
        #conductive 77 - > 4 K
        dq_cond_stands = cf.conductive(A_rod, l_rod, T_heat_shield, T_stage, dt, g10_therm_cond) #the heat shield is not physically connected to these stands but it is assumed to be at the same Temperature
        #and there is no point of using two variables that will just be the same number, this model would break if this assumption were wrong.
        #helium contact
        dq_helium_cond = cf.conductive(A_stage, thickness, T_stage, Tcc, dt, al_therm_cond) #Tc = 4.2 K (temperature of LHe), A = area of plate = pi r^2, l = thickness of plate,
        #total energy
        dQ_stage  = dq_heat_shield_shell + dq_cond_stands - dq_helium_cond
        #change in temp
        dT_stage  = specific_heat(dQ_stage, rho_stage * V_stage, al_specific_heat, T_stag) #should hopefully be negative.

        #energy into the test mass
        dq_rad_4k_shell = cf.rad(e_al, e_al, r_stage, r_tm, T_stage, T_tm, dt, A_stage) #in the case of this eh = ec since its the same material!
        dq_cond_4k_stage = cf.cond(A_test, height, T_stage, T_tm, dt, 1, al_therm_cond) #1 is hard coded here because we are assuming the test mass is one chunk.
        dQ_tm = dq_rad_4k_shell + dq_cond_4k_stage
        dT_tm = specific_heat(dQ_tm, mass, al_specific_heat, al_therm_cond)
        T_tm += dT_tm


def g10_interpolator():
    from scipy.interpolate import interp1d
    T, K = np.loadtxt('g10_therm.txt', delimiter=',', unpack=True)
    interpolator = interp1d(T,K)
    return interpolator

def al_interpolator():
    from scipy.interpolate import interp1d
    T, K = np.loadtxt('al_therm.txt', delimiter=',', unpack=True)
    interpolator = interp1d(T,K)
    return interpolator

def al_specific_heat_interpolator():
    from scipy.interpolate import interp1d
    T, C = np.loadtxt('al_specific_heat.txt', delimiter=',', unpack=True)
    interpolator = interp1d(T, C)
    return interpolator

def omega_calc():
    pass

def integrate_rad():
    #top of the cylinder
    top = omega_t * emissivity * A_circ * (Th**4 - Tc**4)
    #bottom of the cylinder
    bot = omega_b * emissivity * A_circ * (Th**4 - Tc**4)
    #side integrate
    sides = 0
    for hi in height_arr:
        side = omega_h * emissivity * A_h * (Th**4 - Tc**4)
    return top + bot + sides

if __name__ == '__main__':
    #define some of the parameters
    T_heat_shield = 77  # Kelvin
    r_heat_shield = 20*1e-2 #radius of heat shield (m)
    e_stycast = 0.05
    r_stage = np.nan #need my measurements.
    A_stage = np.pi * r_stage**2
    e_al = 0.95 # emissivity of aluminum
    thickness = 5e-3 #thickness of the aluminum plate in (m)
    T_stage = 77 #initial temperature of the stage (K)
    A_rod = np.nan #area of the rod (m)
    l_rod = np.nan #length of the rod (m)
    A_test = np.nan #area of the test mass
    height = np.nan #height of the test mass
    mass = 100e-3 #mass in Kg
    dt = 20 #in seconds
    reach_equilibrium_temp(T_heat_shield, r_heat_shield, e_stycast, A_stage, e_al, thickness, T_stage, Tcc, A_rod, l_rod, A_test, height, mass, nhours, dt)
