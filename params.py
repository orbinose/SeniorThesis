"""
External Parameters
"""

class Bunch(object):
    """
    translates dic['name'] into dic.name 
    """

    def __init__(self, data):
        self.__dict__.update(data)


def cosmo_par():
    nbody_file_in = "/home/jamesdr/scratch/miniramses/output_00011"
    import re
    import numpy as np
    from miniramses.utils.py.miniramses import Info, rd_info
    # Extracting the path using regular expression
    path = re.match(r'(.*/miniramses/)', nbody_file_in).group(1)

    # Extracting the number as an integer
    nout = int(re.match(r'.*/output_(\d+)', nbody_file_in).group(1))
    i = rd_info(nout, path=path)
    par = {
        "z": 0.0,
        "Om": i.omega_m,
        "Ob": 0.049,
        "s8": 0.83,
        "h0": (i.H0/100.0),
        "ns": 0.963,
        "dc": 1.675,
        }
    return Bunch(par)

def baryon_par():
    par = {
        "Mc": 1.0e14,     # beta(M,z): critical mass scale
        "mu": 0.4,        # beta(M,z): critical mass scale
        "nu": 0.0,        # beta(M,c): redshift dependence
        "thej": 4.0,      # ejection factor thej=rej/rvir
        "thco": 0.1,      # core factor thco=rco/rvir
        "alpha": 1.0,     # index in gas profile [default: 1.0]
        "gamma": 2.0,     # index in gas profile [default: 2.0]
        "delta": 7.0,     # index in gas profile [default: 7.0 -> same asympt. behav. than NFWtrunc profile]  
        "rcga": 0.015,    # half-light radius of central galaxy (ratio to rvir)
        "Nstar": 0.04,    # Stellar normalisation param [fstar = Nstar*(Mstar/Mvir)**eta]
        "Mstar": 2.5e11,  # Stellar critical mass [fstar = Nstar*(Mstar/Mvir)**eta]
        "eta": 0.32,      # exponent of total stellar fraction [fstar = Nstar*(Mstar/Mvir)**eta]
        "deta": 0.28,     # exponent of central stellar fraction [fstar = Nstar*(Mstar/Mvir)**(eta+deta)]
        "zeta": 1.38,     # exponent of stellar fraction
        "a_nth": 0.18,    #Non-thermal pressure profile (P_nth = a_nth(b_nth(z))*(r/rvir)^n_nth): normalisation
        "n_nth": 0.8,     #Non-thermal pressure profile: power lae index
        "b_nth": 0.5,     #Non-thermal pressure profile: redshift evolution (b_nth = 0 means no z-evolution)
        }
    return Bunch(par)

def io_files():
    par = {
        "transfct": 'CDM_PLANCK_tk.dat',
        "cosmofct": 'cosmofct.dat',
        "displfct": 'displfct.dat',
        "partfile_in": 'partfile_in.std',
        "partfile_out": 'partfile_out.std',
        "partfile_format": 'tipsy',
        "halofile_in": 'halofile_in.dat',
        "halofile_out": 'halofile_out.dat',
        "halofile_format": 'AHF-ASCII',
        "TNGnumber": 99,   #number of TNG files
    }
    return Bunch(par)

def code_par():
    nbody_file_in = "/home/jamesdr/scratch/miniramses/output_00011"
    import re
    import numpy as np
    from miniramses.utils.py.miniramses import rd_part, Part, rd_info, Info
    # Extracting the path using regular expression
    path = re.match(r'(.*/miniramses/)', nbody_file_in).group(1)

    # Extracting the number as an integer
    nout = int(re.match(r'.*/output_(\d+)', nbody_file_in).group(1))
    p = rd_part(nout, path=path)
    i = rd_info(nout, path=path)
    unit_l = i.unit_l #cm
    unit_d = i.unit_d #g/cm^3
    unit_m = unit_d * (unit_l)**3.0 #h
    unit_m_in_Msol_per_h = unit_m / (1.989e33) * (i.H0/100.0)
    par = {
        "multicomp": True, #individual displacement of collisionless matter and stars/gas?
        "satgal": False, #satellite galaxies treated explicitely (only used in multicomp model).
        "adiab_exp": False, #Adiabatic expasion (used to be turned on, now turned off in default model)
        "kmin": 0.01,
        "kmax": 100.0,
        "rmin": 0.005,
        "rmax": 50.0,
        "Nrbin": 100,
        "rbuffer": 0.0, # buffer size to take care of boundary conditions
        "eps": 4.0,      # truncation factor: eps=rtr/rvir 
        "beta_model": 0, # 0: old model from Schneider+18 1: new model
        "Mhalo_min": 100.0*p.mp[0]*unit_m_in_Msol_per_h,  # Minimum halo mass [Msun/h]
        "disp_trunc": 0.01, # Truncation of displacment funct (disp=0 if disp<disp_trunc) [Mpc/h]
        }
    return Bunch(par)

def sim_par():
    nbody_file_in = "/home/jamesdr/scratch/miniramses/output_00011"
    import re
    from miniramses.utils.py.miniramses import Info, rd_info
    # Extracting the path using regular expression
    path = re.match(r'(.*/miniramses/)', nbody_file_in).group(1)

    # Extracting the number as an integer
    nout = int(re.match(r'.*/output_(\d+)', nbody_file_in).group(1))
    i = rd_info(nout, path=path)
    unit_l = i.unit_l
    par = {
        "Lbox": unit_l / (3.086e24) * ((i.H0/100.0)),   #box size of partfile_in
        "rbuffer": 0.0, #buffer size to take care of boundary conditions
        "Nmin_per_halo": 100,
        "N_chunk": 1      #number of chunks (for multiprocesser: n_core = N_chunk^3)
        }
    return Bunch(par)

def par():
    par = Bunch({"cosmo": cosmo_par(),
        "baryon": baryon_par(),
        "files": io_files(),
        "code": code_par(),
        "sim": sim_par(),
        })
    return par

