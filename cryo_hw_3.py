import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
import sys
import matplotlib.mlab as ml
import matplotlib as mpl

class Lhe_Stage():
    def __init__(self, nbodies, init_temp, specific_heat, radius, height, density, id, therm_cond, emissivity):
        self.nbodies = nbodies
        self.temp_array = np.zeros((nbodies-2))
        self.temp_array[:] = init_temp
        self.top_cap_temp = init_temp
        self.bot_cap_temp = init_temp
        self.spec_heat = specific_heat
        self.density= density
        self.radius = radius
        self.height = height
        self.id = id
        self.mass = np.pi* (radius**2 - id**2) * height * density
        self.emissivity = emissivity
        self.therm_cond = therm_cond
        self.support_T_array = np.zeros((2))
        self.support_T_array[:] = 77
        #2 liter tank -> 0.002 meters only like a 100th of the tank is operational in a given second...
        mhe = 125 * 0.002 / 1000 #assuming 2 liters of liquid helium in the tank at a given time...

        self.max_dq = 23300 * mhe



    def update_temp_array(self, other_rad, stage_height, stage_depth):
        #### the Idea is to do concentric rings of the 4 K stage + its top cap and its bottom cap
        # the last 2 elements in the temperature array are the bot and top cap respectively.
        offset = stage_depth

        #need to start at 1mm or the cond equation will blow up at 0
        bin_heights, h_step = np.linspace(1e-3, self.height, self.nbodies-2, retstep=True)

        bin_other_stage_heights, dh = np.linspace(0, other_rad.length, 1000, retstep=True)
        other_bin_rad, r_step = np.linspace(0, other_rad.radius, 1000, retstep=True)

        four_k_H_col, four_k_H_row = np.meshgrid(bin_heights, bin_heights) #create a two dimensional array that we can integrate

        #the integral needs to be normalized to 1 so for the first step we calculate the contribution from each component
        domega_sides = np.nansum(2 * np.pi * other_rad.radius * h_step * bin_heights / (other_rad.radius**2+bin_heights**2)**(3/2)) #integrate over the sides of the cylinder
        domega_bot   = np.nansum(2 * np.pi * r_step * other_bin_rad / (other_bin_rad**2 + stage_height**2)) #integrate over the bottom of the cylinder
        domega_top   = np.nansum(2 * np.pi * r_step * other_bin_rad / (other_bin_rad**2 + stage_depth**2)) #integrate over the top of the cylinder S
        fa_normalization = domega_sides + domega_bot + domega_top #the final normaliziation

        #vectorized way to calculate the solid angle for each component
        #effectively this is a double integral where we go up in chunks over the 4 K stage and the 77 K stage separately.
        #e.g. for each chunk in the 4 K stage integrate over every chunk in the 77 K stage
        z_dist   = four_k_H_row - bin_heights - offset
        distance = np.sqrt((self.radius - other_rad.radius)**2 + (four_k_H_row - bin_heights- offset)**2)
        rad_area = 2 * np.pi * other_rad.radius * h_step
        domega_sides = np.sum(rad_area * np.abs(z_dist) / distance**(3/2),axis=0)# / fa_normalization
        # domega_top /= fa_normalization; domega_bot #/= fa_normalization
        domega_sides[0] += domega_bot; domega_sides[-1] += domega_top #the bottom and the top of the arrays are assumed to be in thermal equilibrium with the first and last index of the sides array

        dQ_rad = rad_area * 1 / (1/self.emissivity + self.radius/other_rad.radius*(1 /other_rad.emissivity - 1)) *(self.temp_array**4 - other_rad.temperature**4) * 5.67e-8

        ## now to use these to calculate the radiation transfer

        therm_cond_func = self.therm_cond()
        ###therm update
        #the cross sectional area of the 4 K stage is
        cross_sec_area = np.pi * (radius**2 - self.id**2)

        dQ_sides_cond = cross_sec_area / (bin_heights) * (therm_cond_func(self.temp_array) - therm_cond_func(4.5)) * (self.temp_array-4.5)
        dQ = dQ_sides_cond
        support_area = 4 * 4e-2 * 2e-3
        support_length1 = 2.333e-2
        support_length2 = 4.666e-2

        comp_one = support_area / support_length1 * (therm_cond_func(4.5) - therm_cond_func(self.support_T_array[0])) * (4.5 - self.support_T_array[1])
        comp_one -= support_area / support_length2 * (therm_cond_func(77) - therm_cond_func(self.support_T_array[1])) * (77 - self.support_T_array[0])
        comp_two = support_area / support_length2 * (therm_cond_func(4.5) - therm_cond_func(self.support_T_array[1])) * (4.5 - self.support_T_array[1])
        comp_two -= support_area / support_length1 * (therm_cond_func(77) - therm_cond_func(self.support_T_array[0])) * (77 - self.support_T_array[0])
        #


        # dQ_supports  = support_area / support_length * (therm_cond_func(77) - therm_cond_func(self.temp_array)) * (77 - self.temp_array)
        # dQ_support_sum = np.cumsum(dQ_supports)
        dQ_sum = np.cumsum(dQ)
        less_support_1 = np.where(bin_heights < support_length1)[0]
        less_support_2 = bin_heights < support_length2
        great_support_1 = bin_heights > support_length1
        support_2_eq = np.intersect1d(less_support_2, great_support_1)

        absorb_therm_power = (dQ_sum - self.max_dq) > 0
        if np.sum(absorb_therm_power) == 0:
            pass
        elif np.sum(absorb_therm_power) < dQ.size:
            max_dq = self.max_dq
            # if np.sum(absorb_therm_power) > len(less_support_1):
            #     print('support 1')
            #     max_dq = self.max_dq
            # elif np.sum(absorb_therm_power) > len(less_support_2):
            #     print('support 2')
            #     max_dq = self.max_dq - comp_one
            # else:
            #     max_dq = self.max_dq - comp_one - comp_two


            nelements = dQ.size - np.sum(absorb_therm_power) + 1
            absorb_therm_power = (dQ_sum - max_dq) > 0
            last_ele = (dQ_sum - max_dq)[absorb_therm_power][0]
            ind = (dQ_sum - max_dq) < last_ele
            sum_prev = np.sum(dQ[ind])
            last_ele = self.max_dq - sum_prev
            dQ[nelements-1] = last_ele
            dQ[nelements::] = 0

        else:
            dQ[0] = self.max_dq
            dQ[1::] = 0



        dQ = dQ + dQ_rad
        # good_index = np.where()

        m_side_el = h_step * np.pi * (self.radius**2 - self.id**2) * self.density
        c = self.spec_heat()

        dT = dQ / (m_side_el * c(self.temp_array))
        dT_bot = dQ[0] / (h_step * self.radius**2 * self.density)
        dT_top = dQ[-1] / (h_step * self.radius**2 * self.density)

        dT[0] = dT_bot; dT[-1] = dT_top
        self.temp_array -= dT
        index = self.temp_array < 4.5
        self.temp_array[index] = 4.5 #if enough energy is sucked out of the system to create a negative temperature just set it to the floor temperature
        return self.temp_array

class Object():
    def __init__(self, conductor, radiator, dT, cond_dict=None, rad_dict=None, specific_heat=None, mass=None, rho=None):
        if conductor: #check if the object is gaining energy through thermal conducitivty
            self.area = cond_dict['area'] #cross-sectional area
            self.length = cond_dict['length'] #length of the rod
            self.thermal_cond_func = cond_dict['thermal_cond_func'] #thermal conducitivty function
            self.temperature = cond_dict['temperature']#temprature of the rods
            self.num_objects = cond_dict['num_objects']#number of the rods that are thermally conductive
        if radiator: #check if the object is radiating away thermal energy
            self.radiating_area = rad_dict['area']  #the area that is emitting
            self.emissivity = rad_dict['emissivity']  #emissivity of the radiator
            self.radius = rad_dict['radius'] #here this is the radius of a cylinder, in this problem we are assuming all radiators are cylinders, but other geometries exist
            self.temperature = rad_dict['temperature'] #temperature of the object
            self.length = rad_dict['height']
        if dT: #check if we have to calculate a chance in tempeerature through gaining thermal energy
            self.specific_heat_func = specific_heat
            self.mass = mass

    def conductive(self, other, dt):
        k_h = self.thermal_cond_func(self.temperature)
        k_c = self.thermal_cond_func(other.temperature)
        return self.num_objects * self.area / self.length * (k_h - k_l)* (self.temperature - other.temperature) * dt

    def rad(self, other, dt):
        #this function assumes competing radiation from two cylindrical radiators
        return self.radiating_area *  1 / (1/self.emissivity + self.radius/other.radius*(1 /other.emissivity - 1)) *(self.temperature**4 - other.temperature**4) * dt * 5.67e-8

    def specific_heat(dQ):
        c = self.specific_heat(self.temperature)
        dT = dQ / (self.mass*c)


#schematic of objects....


# 77 K heat shield -> rad/cond of 4 K stage
# 4 K stage -> rad/cond holding apparatus
# 4 K stage -> rad test mass
# holding apparatus -> cond test_mass

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

    holder_normalizing_factor = holder_omega_calc()
    test_m_normalizing_factor = test_mass_omega_calc()
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

        #energy into the test mass holder

        #replace dq_rad_4k_shell with the numerical integration...

        #for this one can't have the surfaces touching doing rad cuz that will blow up value
        #need to separate into two, outer_ring and inner_ring integrations where the radial distance component is (r_dist - od) and (r_dist - id) respectively.
        dq_rad_4k_shell = cf.rad(e_al, e_al, r_stage, r_holder, T_stage, T_holder, dt, A_stage) #in the case of this eh = ec since its the same material!
        dq_cond_4k_stage = cf.cond(A_holder, height_holder, T_stage, T_holder, dt, 1, steel_therm_cond) #1 is hard coded here because we are assuming the test mass is one chunk.
        dQ_holder = dq_rad_4k_shell + dq_cond_4k_stage
        dT_holder = specific_heat(dQ_tm, mass, al_specific_heat, steel_therm_cond)


        dQ_test_mass = None #numerical integrator...
        dQ_test_mass

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

def test_mass_omega_calc(cylinder_height, cylinder_radius, sep_dist):
    cylinder_height = 30e-2
    cylinder_radius = 20e-2
    separataion_dist = 2e-2
    bin_height, step = np.linspace(0, 30e-2, 1000, retstep=True)
    domega = 0
    curr_height = 0
    dA = 0
    for hi in bin_height:
        curr_height += step
        dA += 2 * np.pi * cylinder_radius * step
        domega += 2 * np.pi * cylinder_radius * step * curr_height / (cylinder_radius**2 + curr_height**2)**(3/2)
    bin_radius, r_step = np.linspace(0, 20e-2, 1000, retstep=True)
    domega_top_cap = 0
    curr_r = 0
    dA = 0
    for ri in bin_radius:
        curr_r += r_step
        dA += 2 * np.pi * curr_r * r_step

        domega_top_cap += 2 * np.pi * curr_r * r_step / (curr_r**2+cylinder_height**2)
    sum = domega_top_cap +  domega #no dq bottom since it is ontop of a holding apparatus that is larger than the sample
    return sum

def holder_omega_calc(cylinder_height, cylinder_radius, holder_id, holder_od):
    bin_heights, h_step = np.linspace(0, cylinder_height, 1000, retstep = True)
    bin_radii, r_step = np.linspace(0, cylinder_radius, 1000, retstep=True)

    domega_sides = np.nansum(2 * np.pi * cylinder_radius * h_step * bin_heights / (cylinder_radius**2+bin_heights**2)**(3/2))
    id_radii = bin_radii < holder_id
    domega_id    = np.nansum(2 * np.pi * r_step * bin_radii[id_radii] / (bin_radii[id_radii]**2))
    od_radii = bin_radii > holder_od
    domega_od    = np.nansum(2 * np.pi * r_step * bin_radii[od_radii] / (bin_radii[od_radii])**2)
    domega_bot = domega_id + domega_od
    domega_top   = np.nansum(bin_radii * 2 * np.pi * r_step / (bin_radii**2 + cylinder_height**2))
    return domega_top + domega_bot + domega_sides

def integrate_rad_testmass(cylinder_height, cylinder_radius, sep_dist, emissivity, Th, Tc, normalizing_factor):
    #top of the cylinder
    bin_heights, h_step = np.linspace(0, cylinder_height, 1000, retstep = True)
    bin_radii, r_step = np.linspace(0, cylinder_radius, 1000, retstep=True)

    #integrate sides
    curr_height = 0
    dQ = 0
    dA = 2 * np.pi * cylinder_radius * h_step
    dQ_sides = np.sum(dA * bin_heights / (cylinder_radius**2 + bin_heights**2)**(3/2) / normalizing_factor * 1 / (1/emissivity + cylinder_radius/5e-3*(1 /emissivity - 1))*(Th**4 - Tc**4) * 5.67e-8 )

    dQ_top   = np.sum(bin_radii * 2 * np.pi * r_step / (bin_radii**2 + cylinder_height**2)) / normalizing_factor
    return dQ_sides + dQ_top

def integrate_rad_holder(holder_height, holder_radius, emissivity, Th, Tc, norm_fac):
    normalizing_factor = test_omega_calc(cylinder_height, cylinder_radius, sep_dist)
    bin_heights, h_step = np.linspace(0, cylinder_height, 1000, retstep = True)
    bin_radii, r_step = np.linspace(0, cylinder_radius, 1000, retstep=True)

    #integrate sides
    curr_height = 0
    dQ = 0
    dA = 2 * np.pi * cylinder_radius * h_step
    dQ_sides = np.sum(dA * bin_heights / (cylinder_radius**2 + bin_heights**2)**(3/2) / normalizing_factor * 1 / (1/emissivity + cylinder_radius/5e-3*(1 /emissivity - 1))*(Th**4 - Tc**4) * 5.67e-8 )
    dQ_top   = np.sum(bin_radii * 2 * np.pi * r_step / (bin_radii**2 + cylinder_height**2)) / normalizing_factor
    return dQ_sides + dQ_top


def animate_movie(image_set, time, avg_T):
    plt.rcParams.update({'font.size': 28})
    fig, axes = plt.subplot_mosaic(
    """
    AAAAABBBBBD
    """,
    constrained_layout=True, figsize=(16,12)
    )
    plt.rcParams.update({'font.size': 28})
    # fig, axes = plt.subplots(1,5, figsize=(16,12))
    camera = Camera(fig)

    cmap = mpl.cm.coolwarm
    norm = mpl.colors.Normalize(vmin=4.5, vmax=77)
    cb1 = mpl.colorbar.ColorbarBase(axes['D'], cmap=cmap, norm=norm, orientation='vertical')
    axes['A'].axvline(time[i], color='black')
    axes['A'].plot(time[1::], avg_T[1::])
    axes['A'].set_xlabel('time since start [Min]')
    axes['A'].set_ylabel('Average temperature of LHe stage')

    for i in range(image_set.shape[0]):
        im1 = axes['B'].imshow(image_set[i], origin='lower', cmap=cmap, norm=norm)
        cb1 = mpl.colorbar.ColorbarBase(axes['D'], cmap=cmap, norm=norm, orientation='vertical', label='Temperature [K]')

        camera.snap()
    anim = camera.animate(blit=True, interval=500)
    anim.save('4k_stage_cooldown.gif')

if __name__ == '__main__':
    ln2_height = 25e-2; ln2_radius = 15e-2
    ln2_rad_dict = {'area': 2*np.pi * ln2_radius * ln2_height + 2 * np.pi * ln2_radius**2, 'emissivity': 0.05,'radius' : ln2_radius, 'temperature': 77, 'height': ln2_height}
    ln2_stage = Object(False, True, False, rad_dict = ln2_rad_dict)

    ##
    nbodies = 17; init_temp = 77; radius=10e-2; height=20e-2; density=2710; id_radius = 9.9e-2
    four_k_stage = Lhe_Stage(nbodies, init_temp, al_specific_heat_interpolator, radius, height, density, id_radius, al_interpolator, 0.95)

    array = np.zeros((25,16))

    array[:,:] = np.nan
    #hard coded but making fun images is kind of annoying and its getting beyond the scope of this project to do this part in a scalable robust way...
    array[0,:] = 77
    array[:,0] = 77
    array[24,:] = 77
    array[:,15] = 77
    array[3:6,3:13] = 4.5
    array[6:21,3] = 77
    array[6:21,12] = 77
    array[21,3:13] = 77

    array_set = np.zeros((251,25,16))
    array_set[0,:] = array
    ind = 1
    time_list = [0]
    avg_T = [77]
    time = 0

    for i in range(1000):
        temp_arr = four_k_stage.update_temp_array(ln2_stage, 2.5e-2, 2.5e-2)
        if i % 4 == 0:
            array[6:21,3] = temp_arr
            array[6:21,12] = temp_arr
            array[21,3:13] = temp_arr[-1]
            array_set[ind,:] = array
            ind += 1
            time = time + 5/60
            time_list.append(time)
            avg_T.append(np.mean(temp_arr))
    animate_movie(array_set, time_list,avg_T)

    exit()
    #
    # lhe_height = 15e-2; lhe_radius = 5e-2; lhe_id_radius = 3e-2
    #
    # lhe_cond_dict = {'area': np.pi*lhe_radius**2 - np.pi * lhe_id_radius**2,
    #                  'length': lhe_height,
    #                  'thermal_cond_func': al_therm_cond}
    #                  'temperature': 77, #starting at 77 K then cools down to 4 K
    #                  'num_objects': 4}
    # lhe_density = 900 #some number
    # lhe_volume  = (np.pi * lhe_radius**2 - np.pi*lhe_id_radius**2)* lhe_height
    #
    #
    # #lhe_stage does gain heat through conduction but in this instance we consider a graident rather than the generic calculation
    # lhe_stage = Object(False, False, True, specific_heat = al_specific_heat_interpolator, mass=lhe_volume * lhe_density)
    # #even though the Lhe stage radiates we treat it in a special way since we are looking at the temperature gradient whereas ln2 stage is fixed at 77
    #
    #

    #
    #
    # holder_od = 10e-3; holder_id = 2e-3; holder_height = 2e-2
    # hold_vol = (np.pi * holder_od**2 - np.pi*holder_id**2) * holder_height
    # test_holder = Object(False, False, True, specific_heat=steel_specfic_heat, mass=hold_vol * hold_density)
    #
    #
    # tm_radius = 0
    # test_mass_cond_dict = {'area': np.pi * tm_radius**2,
    #                        'length': tm_length,
    #                        'thermal_cond_func': al_therm_cond,
    #                        'temperature': 77,
    #                        'num_objects':1}
    # test_mass = Object(True, False True, specific_heat=al_specific_heat_interpolator, mass=100e-3)
    #
    # # test_omega_calc()
    # holder_od = 10e-3; holder_id = 2e-3
    # cylinder_height = 30e-2; cylinder_radius = 20e-2
    # normalizing_factor = holder_omega_calc(cylinder_height, cylinder_radius, holder_id, holder_od)
    # print(normalizing_factor)
    # normalizing_factor = test_omega_calc(cylinder_height, cylinder_radius, 2e-2)

    integrate_rad(cylinder_height, cylinder_radius, 2e-2, 0.05, 77, 4, normalizing_factor)
    exit()
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
