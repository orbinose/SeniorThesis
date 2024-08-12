"""
CALCULATE DISPLACEMENT FUNCTION FOR A GRID OF M AND C
PRINT INFORMATION INTO TEMPORARY FILE

"""

import numpy as np
from scipy import spatial
from scipy.interpolate import splrep,splev
from numpy.lib.recfunctions import append_fields

import schwimmbad

from .constants import *
from .profiles import *


"""
READING/WRITING FILES
"""

def read_nbody_file(param):

    """
    Read in N-body output, adopt units, and build chunks 
    (for multi-processor mode)
    Only supports tispy file format for the moment
    """

    nbody_file_in = param.files.partfile_in
    nbody_file_format = param.files.partfile_format
    Lbox   = param.sim.Lbox
    N_chunk = param.sim.N_chunk
    L_chunk = Lbox/N_chunk

    #read in file
    if (nbody_file_format=='tipsy'):
        import numpy as np
        try:
            f = open(nbody_file_in, 'r')
        except IOError:
            print('IOERROR: N-body tipsy file does not exist!')
            print('Define par.files.partfile_in = "/path/to/file"')
            exit()

        #header
        p_header_dt = np.dtype([('a','>d'),('Npart','>i'),('dim','>i'),('Ngas','>i'),('Ndm','>i'),('Nstar','>i'),('buffer','>i')])
        p_header = np.fromfile(f, dtype=p_header_dt, count=1, sep='')

        #particles
        p_dt = np.dtype([('mass','>f'),("x",'>f'),("y",'>f'),("z",'>f'),("vx",'>f'),("vy",'>f'),("vz",'>f'),("eps",'>f'),("phi",'>f')])
        p_dm = np.fromfile(f, dtype=p_dt, count=int(p_header['Npart']), sep='')

        #convert to Mpc/h
        p_dm['x']=Lbox*(p_dm['x']+0.5)
        p_dm['y']=Lbox*(p_dm['y']+0.5)
        p_dm['z']=Lbox*(p_dm['z']+0.5)

        #convert to Mstar/h
        print("Om = ", np.sum(p_dm['mass']))
        p_dm['mass'] = p_dm['mass'] * rhoc_of_z(param)*Lbox**3
        
        print('Reading tipsy-file done!')

    elif (nbody_file_format=='TNG'):

        import illustris_python as il
        basePath = nbody_file_in
        TNGnumber = int(param.files.TNGnumber)

        dm_pos = il.snapshot.loadSubset(basePath,TNGnumber,'dm',['Coordinates'],float32=True)/1000.0  #Mpc/h
        dm_vel = il.snapshot.loadSubset(basePath,TNGnumber,'dm',['Velocities'],float32=True)
        dm_pot = il.snapshot.loadSubset(basePath,TNGnumber,'dm',['Potential'],float32=True)

        header = il.groupcat.loadHeader(basePath,TNGnumber)
        print(header)
        mass = (header['Omega0']/len(dm_pos[:,0])).astype(np.float32)
        dm_mass = np.full(len(dm_pos[:,0]),mass)
        
        #header
        p_header_dt = np.dtype([('a','>d'),('Npart','>i'),('dim','>i'),('Ngas','>i'),('Ndm','>i'),('Nstar','>i'),('buffer','>i')])
        p_header = np.zeros(1,dtype=p_header_dt)
        p_header['Npart'] = len(dm_pos[:,0])
        p_header['Ndm'] = len(dm_pos[:,0])
        p_header['dim'] = int(3)        
        
        #particles
        p_dt = np.dtype([('mass','>f'),("x",'>f'),("y",'>f'),("z",'>f'),("vx",'>f'),("vy",'>f'),("vz",'>f'),("eps",'>f'),("phi",'>f')])
        p_dm = np.zeros(p_header['Npart'],dtype=p_dt)
        p_dm['mass'] = dm_mass
        p_dm['x'], p_dm['y'], p_dm['z']    = dm_pos[:,0], dm_pos[:,1], dm_pos[:,2]
        p_dm['vx'], p_dm['vy'], p_dm['vz'] = dm_vel[:,0], dm_vel[:,1], dm_vel[:,2]
        p_dm['phi'] = dm_pot

        print('Reading hdf5 file from IllustrisTNG done!')

    elif (nbody_file_format=='gadget'):

        print('Reading gadget files not implemented. Exit!')
        exit()

    elif (nbody_file_format=='ramses'):
        import re
        import numpy as np
        from astropy.io import ascii
        from miniramses.utils.py.miniramses import rd_part, Part, Info, rd_info

        # Extracting the path using regular expression
        path = re.match(r'(.*/miniramses/)', nbody_file_in).group(1)

        # Extracting the number as an integer
        nout = int(re.match(r'.*/output_(\d+)', nbody_file_in).group(1))
        
        i = rd_info(nout,path=path)
        
        p = rd_part(nout,path=path)
        
        unit_l = i.unit_l #cm
        unit_d = i.unit_d #g/cm^3
        unit_t = i.unit_t #s
        
        unit_v = unit_l/unit_t #cm/s
        unit_m = unit_d * (unit_l)**3.0 #g
        
        #We convert to Mpc/h, Msol/h, and km/s
        
        unit_l_in_Mpc_per_h = unit_l / (3.086e24) * param.cosmo.h0
        
        unit_m_in_Msol_per_h = unit_m / (1.989e33) * param.cosmo.h0
        
        unit_v_in_km_per_s = unit_v / (1e5)
        
        p_header_dt = np.dtype([('a','>d'),('Npart','>i'),('dim','>i'),('Ngas','>i'),('Ndm','>i'),('Nstar','>i'),('buffer','>i')])
        p_header = np.zeros(1,dtype=p_header_dt)
        p_header['Npart'] = p.np
        p_header['Ndm'] = p.np
        p_header['dim'] = i.ndim
        
        p_dt = np.dtype([('mass','>f'),("x",'>f'),("y",'>f'),("z",'>f'),("vx",'>f'),("vy",'>f'),("vz",'>f'),("eps",'>f'),("phi",'>f')])
        p_dm = np.zeros(p.np,dtype=p_dt)
        p_dm['mass'] = p.mp * unit_m_in_Msol_per_h
        p_dm['x'], p_dm['y'], p_dm['z']    = p.xp[0,:] * unit_l_in_Mpc_per_h, p.xp[1,:] * unit_l_in_Mpc_per_h, p.xp[2,:] * unit_l_in_Mpc_per_h
        p_dm['vx'], p_dm['vy'], p_dm['vz'] = p.vp[0,:] * unit_v_in_km_per_s, p.vp[1,:] * unit_v_in_km_per_s, p.vp[2,:] * unit_v_in_km_per_s
        
    else:
        print('Unknown file format. Exit!')
        exit()

    #split particles into chunks
    p_dm_list   = []
    for x_min in np.linspace(0,Lbox-L_chunk,N_chunk):
        x_max = x_min + L_chunk
        if (x_max == Lbox):
            x_max = 1.00001*x_max
        for y_min in np.linspace(0,Lbox-L_chunk,N_chunk):
            y_max =y_min + L_chunk
            if (y_max == Lbox):
                    y_max = 1.00001*y_max
            for z_min in np.linspace(0,Lbox-L_chunk,N_chunk):
                z_max = z_min + L_chunk
                if (z_max == Lbox):
                        z_max = 1.00001*z_max
                ID_dm = np.where((p_dm['x']>=x_min) & (p_dm['x']<x_max) & (p_dm['y']>=y_min) & (p_dm['y']<y_max) & (p_dm['z']>=z_min) & (p_dm['z']<z_max))
                p_dm_list += [p_dm[ID_dm]]

    #check if number of particles is still the same 
    pl = 0
    for pp in p_dm_list:
        pl += len(pp)
    if (pl != len(p_dm)):
        print('Chunking: particle number not conserved! Exit.')
        exit()

    return p_header, p_dm_list


def write_nbody_file(p_header,p_gas,p_dm,p_star,param):

    """
    Combine chunks and write N-body outputs with displaced 
    particles. Only tipsy file format for the moment.
    p_list = list of p with legth = Nchunk 
    """

    nbody_file_out = param.files.partfile_out
    nbody_file_format = param.files.partfile_format
    Lbox = param.sim.Lbox
    N_chunk = param.sim.N_chunk

    #correct for periodic boundaries
    p_gas['x'][p_gas['x']>Lbox] -= Lbox
    p_gas['x'][p_gas['x']<0.0]  += Lbox
    p_gas['y'][p_gas['y']>Lbox] -= Lbox
    p_gas['y'][p_gas['y']<0.0]  += Lbox
    p_gas['z'][p_gas['z']>Lbox] -= Lbox
    p_gas['z'][p_gas['z']<0.0]  += Lbox
    p_dm['x'][p_dm['x']>Lbox] -= Lbox
    p_dm['x'][p_dm['x']<0.0]  += Lbox
    p_dm['y'][p_dm['y']>Lbox] -= Lbox
    p_dm['y'][p_dm['y']<0.0]  += Lbox
    p_dm['z'][p_dm['z']>Lbox] -= Lbox
    p_dm['z'][p_dm['z']<0.0]  += Lbox
    p_star['x'][p_star['x']>Lbox] -= Lbox
    p_star['x'][p_star['x']<0.0]  += Lbox
    p_star['y'][p_star['y']>Lbox] -= Lbox
    p_star['y'][p_star['y']<0.0]  += Lbox
    p_star['z'][p_star['z']>Lbox] -= Lbox
    p_star['z'][p_star['z']<0.0]  += Lbox
    
    #write output
    #if (nbody_file_format=='tipsy'):
    if (nbody_file_format=='tipsy' or nbody_file_format=='TNG'):

        try:
            f = open(nbody_file_out, 'wb')
        except IOError:
            print('IOERROR: Path to output file does not exist!')
            print('Define par.files.partfile_out = "/path/to/file"')
            exit()

        #back to tipsy position units
        p_gas['x']  = (p_gas['x']/Lbox-0.5)
        p_gas['y']  = (p_gas['y']/Lbox-0.5)
        p_gas['z']  = (p_gas['z']/Lbox-0.5)
        p_dm['x']   = (p_dm['x']/Lbox-0.5)
        p_dm['y']   = (p_dm['y']/Lbox-0.5)
        p_dm['z']   = (p_dm['z']/Lbox-0.5)
        p_star['x'] = (p_star['x']/Lbox-0.5)
        p_star['y'] = (p_star['y']/Lbox-0.5)
        p_star['z'] = (p_star['z']/Lbox-0.5)

        #back to tipsy mass units
        p_gas['mass'] = p_gas['mass']/rhoc_of_z(param)/Lbox**3
        p_dm['mass']  = p_dm['mass']/rhoc_of_z(param)/Lbox**3
        p_star['mass'] = p_star['mass']/rhoc_of_z(param)/Lbox**3

        print("Om = ", np.sum(p_gas['mass'])+np.sum(p_dm['mass'])+np.sum(p_star['mass']))
        
        p_gas_type  = np.dtype([('mass','>f4'),('x', '>f4'),('y', '>f4'),('z', '>f4'),('vx', '>f4'),('vy', '>f4'),\
                               ('vz', '>f4'),('rho','>f4'),('temp','>f4'),('hsmooth','>f4'),('metals','>f4'),('phi','>f4')])
        p_dm_type   = np.dtype([('mass','>f4'),("x",'>f4'),("y",'>f4'),("z",'>f4'),("vx",'>f4'),("vy",'>f4'),("vz",'>f4'),\
                                ("eps",'>f4'),("phi",'>f4')])
        p_star_type = np.dtype([('mass','>f4'),('x', '>f4'),('y', '>f4'),('z', '>f4'),('vx', '>f4'),('vy', '>f4'),\
                                 ('vz', '>f4'),('metals','>f4'),('tform','>f4'),('eps','>f4'),('phi','>f4')])

        p_gas_mtype  = np.zeros(p_header['Ngas'],dtype=p_gas_type)
        p_dm_mtype   = np.zeros(p_header['Ndm'],dtype=p_dm_type)
        p_star_mtype = np.zeros(p_header['Nstar'],dtype=p_star_type)

        p_gas_mtype['mass'] = p_gas['mass'].astype(np.float32)
        p_gas_mtype['temp'] = p_gas['temp'].astype(np.float32)
        p_gas_mtype['x'] = p_gas['x'].astype(np.float32)
        p_gas_mtype['y'] = p_gas['y'].astype(np.float32)
        p_gas_mtype['z'] = p_gas['z'].astype(np.float32)
        p_gas_mtype['vx'] = p_gas['vx'].astype(np.float32)
        p_gas_mtype['vy'] = p_gas['vy'].astype(np.float32)
        p_gas_mtype['vz'] = p_gas['vz'].astype(np.float32)
        p_gas_mtype['phi'] = p_gas['phi'].astype(np.float32)

        p_dm_mtype['mass'] = p_dm['mass'].astype(np.float32)
        p_dm_mtype['x'] = p_dm['x'].astype(np.float32)
        p_dm_mtype['y'] = p_dm['y'].astype(np.float32)
        p_dm_mtype['z'] = p_dm['z'].astype(np.float32)
        p_dm_mtype['vx'] = p_dm['vx'].astype(np.float32)
        p_dm_mtype['vy'] = p_dm['vy'].astype(np.float32)
        p_dm_mtype['vz'] = p_dm['vz'].astype(np.float32)
        p_dm_mtype['phi'] = p_dm['phi'].astype(np.float32)

        p_star_mtype['mass'] = p_star['mass'].astype(np.float32)
        p_star_mtype['x'] = p_star['x'].astype(np.float32)
        p_star_mtype['y'] = p_star['y'].astype(np.float32)
        p_star_mtype['z'] = p_star['z'].astype(np.float32)
        p_star_mtype['vx'] = p_star['vx'].astype(np.float32)
        p_star_mtype['vy'] = p_star['vy'].astype(np.float32)
        p_star_mtype['vz'] = p_star['vz'].astype(np.float32)
        p_star_mtype['phi'] = p_star['phi'].astype(np.float32)

        del p_gas
        del p_dm
        del p_star

        print("Total number of star particles", len(p_star_mtype['mass']))
        
        p_header.tofile(f,sep='')
        p_gas_mtype.tofile(f,sep='')
        p_dm_mtype.tofile(f,sep='')
        p_star_mtype.tofile(f,sep='')
        
    elif (nbody_file_format=='gadget'):
        print('Writing gadget files not implemented. Exit!')
        exit()
        
    elif (nbody_file_format=='ramses'):
        try:
            f = open(nbody_file_out, 'wb')
        except IOError:
            print('IOERROR: Path to output file does not exist!')
            print('Define par.files.partfile_out = "/path/to/file"')
            exit()

        #back to tipsy position units
        p_gas['x']  = (p_gas['x']/Lbox-0.5)
        p_gas['y']  = (p_gas['y']/Lbox-0.5)
        p_gas['z']  = (p_gas['z']/Lbox-0.5)
        p_dm['x']   = (p_dm['x']/Lbox-0.5)
        p_dm['y']   = (p_dm['y']/Lbox-0.5)
        p_dm['z']   = (p_dm['z']/Lbox-0.5)
        p_star['x'] = (p_star['x']/Lbox-0.5)
        p_star['y'] = (p_star['y']/Lbox-0.5)
        p_star['z'] = (p_star['z']/Lbox-0.5)

        #back to tipsy mass units
        p_gas['mass'] = p_gas['mass']/rhoc_of_z(param)/Lbox**3
        p_dm['mass']  = p_dm['mass']/rhoc_of_z(param)/Lbox**3
        p_star['mass'] = p_star['mass']/rhoc_of_z(param)/Lbox**3

        print("Om = ", np.sum(p_gas['mass'])+np.sum(p_dm['mass'])+np.sum(p_star['mass']))
        
        p_gas_type  = np.dtype([('mass','>f4'),('x', '>f4'),('y', '>f4'),('z', '>f4'),('vx', '>f4'),('vy', '>f4'),\
                               ('vz', '>f4'),('rho','>f4'),('temp','>f4'),('hsmooth','>f4'),('metals','>f4'),('phi','>f4')])
        p_dm_type   = np.dtype([('mass','>f4'),("x",'>f4'),("y",'>f4'),("z",'>f4'),("vx",'>f4'),("vy",'>f4'),("vz",'>f4'),\
                                ("eps",'>f4'),("phi",'>f4')])
        p_star_type = np.dtype([('mass','>f4'),('x', '>f4'),('y', '>f4'),('z', '>f4'),('vx', '>f4'),('vy', '>f4'),\
                                 ('vz', '>f4'),('metals','>f4'),('tform','>f4'),('eps','>f4'),('phi','>f4')])

        p_gas_mtype  = np.zeros(p_header['Ngas'],dtype=p_gas_type)
        p_dm_mtype   = np.zeros(p_header['Ndm'],dtype=p_dm_type)
        p_star_mtype = np.zeros(p_header['Nstar'],dtype=p_star_type)

        p_gas_mtype['mass'] = p_gas['mass'].astype(np.float32)
        p_gas_mtype['temp'] = p_gas['temp'].astype(np.float32)
        p_gas_mtype['x'] = p_gas['x'].astype(np.float32)
        p_gas_mtype['y'] = p_gas['y'].astype(np.float32)
        p_gas_mtype['z'] = p_gas['z'].astype(np.float32)
        p_gas_mtype['vx'] = p_gas['vx'].astype(np.float32)
        p_gas_mtype['vy'] = p_gas['vy'].astype(np.float32)
        p_gas_mtype['vz'] = p_gas['vz'].astype(np.float32)
        p_gas_mtype['phi'] = p_gas['phi'].astype(np.float32)

        p_dm_mtype['mass'] = p_dm['mass'].astype(np.float32)
        p_dm_mtype['x'] = p_dm['x'].astype(np.float32)
        p_dm_mtype['y'] = p_dm['y'].astype(np.float32)
        p_dm_mtype['z'] = p_dm['z'].astype(np.float32)
        p_dm_mtype['vx'] = p_dm['vx'].astype(np.float32)
        p_dm_mtype['vy'] = p_dm['vy'].astype(np.float32)
        p_dm_mtype['vz'] = p_dm['vz'].astype(np.float32)
        p_dm_mtype['phi'] = p_dm['phi'].astype(np.float32)

        p_star_mtype['mass'] = p_star['mass'].astype(np.float32)
        p_star_mtype['x'] = p_star['x'].astype(np.float32)
        p_star_mtype['y'] = p_star['y'].astype(np.float32)
        p_star_mtype['z'] = p_star['z'].astype(np.float32)
        p_star_mtype['vx'] = p_star['vx'].astype(np.float32)
        p_star_mtype['vy'] = p_star['vy'].astype(np.float32)
        p_star_mtype['vz'] = p_star['vz'].astype(np.float32)
        p_star_mtype['phi'] = p_star['phi'].astype(np.float32)

        del p_gas
        del p_dm
        del p_star

        print("Total number of star particles", len(p_star_mtype['mass']))
        
        p_header.tofile(f,sep='')
        p_gas_mtype.tofile(f,sep='')
        p_dm_mtype.tofile(f,sep='')
        p_star_mtype.tofile(f,sep='')

    else:
        print('Unknown file format. Exit!')
        exit()


def read_halo_file(param):

    """
    Read in halo file, adopt units, build buffer around 
    chunks (for multi-processor mode)
    Select for hosts with more than 100 particles.
    Restricted to AHF for the moment.
    """

    #read files 
    halo_file_in = param.files.halofile_in
    halo_file_format = param.files.halofile_format
    Nmin = param.sim.Nmin_per_halo
    Lbox = param.sim.Lbox

    if (halo_file_format=='AHF-ASCII'):
        import numpy as np
        try:
            names = "ID,IDhost,Mvir,x,y,z,rvir,cvir"
            h = np.genfromtxt(halo_file_in,usecols=(0,1,3,5,6,7,11,42),comments='#',dtype=None,names=names)
        except IOError:
            print('IOERROR: AHF-ASCII file does not exist!')
            print('Define par.files.halofile_in = "/path/to/file"')
            exit()

        #adopt units
        h['x']    = h['x']/1000.0
        h['y']    = h['y']/1000.0
        h['z']    = h['z']/1000.0
        h['rvir'] = h['rvir']/1000.0

        #select haloes with reasonable concentration
        gID  = np.where(h['cvir'] > 0)
        h = h[gID]

        #select haloes abov Mmin
        h = h[h['Mvir']>param.code.Mhalo_min]        
        print('Nhalo = ',len(h['Mvir']))

    elif (halo_file_format=='ROCKSTAR-NPY'):

        #h_dt = np.dtype([('ID', '<i8'), ('IDhost', '<i8'), ('x', '<f8'), ('y', '<f8'),
        #                 ('z', '<f8'), ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'),
        #                 ('Mv', '<f8'), ('mpeak', '<f8'), ('vmp', '<f8'), ('r', '<f8'),
        #                 ('sm', '<f8'), ('icl', '<f8'), ('sfr', '<f8'), ('ssfr', '<f8'),
        #                 ('pid', '<f8'), ('Mvir', '<f8'), ('rvir', '<f8'),
        #                 ('rs_hlist', '<f8'), ('scale_half_mass', '<f8'), ('scale_last_mm', '<f8'),
        #                 ('m200b_hlist', '<f8'), ('m200c_hlist', '<f8'), ('gamma_inst', '<f8'),
        #                 ('gamma_100myr', '<f8'), ('gamma_1tdyn', '<f8'), ('gamma_2tdyn', '<f8'),
        #                 ('gamma_mpeak', '<f8'), ('vmax_mpeak', '<f8'), ('halo_hostid', '<i8'),
        #                 ('mhalo_host', '<f8'), ('mask_central', '?'), ('mtot_galaxy', '<f8'),
        #                 ('mstar_mhalo', '<f8'), ('logms_gal', '<f8'), ('logms_icl', '<f8'),
        #                 ('logms_tot', '<f8'), ('logms_halo', '<f8'), ('logmh_vir', '<f8'),
        #                 ('logmh_peak', '<f8'), ('logmh_host', '<f8')])

        try:
            names = "ID,IDhost,Mvir,x,y,z,rvir"
            h = np.genfromtxt(halo_file_in,usecols=(0,1,17,2,3,4,18),comments='#',dtype=None,names=names)
        except IOError:
            print('IOERROR: AHF-ASCII file does not exist!')
            print('Define par.files.halofile_in = "/path/to/file"')
            exit()

        h = append_fields(h, 'cvir', h['rvir']/h['rs_hlist'])

        #adopt units
        h['rvir'] = h['rvir']/1000
        h['rs_hlist'] = h['rs_hlist']/1000
        
        #select haloes with reasonable concentration
        gID  = np.where(np.isfinite(h['cvir']))
        h = h[gID]

        #select haloes above Mmin
        h = h[h['Mvir']>param.code.Mhalo_min]
        print('Nhalo = ',len(h['Mvir']))

    elif (halo_file_format=='TNG'):

        """
        In the TNG/Subfind language there are (FoF) groups and subhalos. The central/largest subhalo
        corresponds to the main halo. Its mass needs to be calculated from the FoF group (as the SubhaloMass
        excludes the mass of other Subhalos) 
        """
        
        import illustris_python as il
        basePath = halo_file_in
        TNGnumber = int(param.files.TNGnumber)

        h_header = il.groupcat.loadHeader(basePath,TNGnumber)
        
        #Subhalo position
        h_pos  = il.groupcat.loadSubhalos(basePath,TNGnumber,['SubhaloPos'])/1000.0   # [Mpc/h]

        #Subhalo mass (for main haloes its Mvir_crit for subhaloes its the total mass of all particles)
        h_Mvir = il.groupcat.loadSubhalos(basePath,TNGnumber,['SubhaloMass'])*1e10   # [Msun/h]
        h_Mvir[il.groupcat.loadHalos(basePath,TNGnumber,fields=['GroupFirstSub'])] = il.groupcat.loadHalos(basePath,TNGnumber,['Group_M_Crit200'])*1e10

        #Subhalo radius (for main haloes its rvir_crit, for subhaloes its the half-mass-radius)
        h_rvir = il.groupcat.loadSubhalos(basePath,TNGnumber,['SubhaloVmaxRad'])/1000.0   # [Mpc/h] 
        h_rvir[il.groupcat.loadHalos(basePath,TNGnumber,fields=['GroupFirstSub'])] = il.groupcat.loadHalos(basePath,TNGnumber,['Group_R_Crit200'])/1000.0
        
        #host halo (=-1)
        h_IDhost = il.groupcat.loadSubhalos(basePath,TNGnumber,['SubhaloGrNr'])
        h_IDhost[il.groupcat.loadHalos(basePath,TNGnumber,fields=['GroupFirstSub'])] = -1

        #IDs of (sub-) haloes
        h_ID = np.arange(0,len(il.groupcat.loadSubhalos(basePath,TNGnumber,['SubhaloCM'])),1)

        
        #assuming concentration-mass relation from Dutton and Maccio (1402.7073)
        zz    = h_header['Redshift']
        AA    = 0.520 + (0.905-0.520)*np.exp(-0.617*zz**1.21)
        BB    = -0.101 + 0.026*zz
        h_cvir = 10**AA * (h_Mvir/1e12)**BB

        "ID,IDhost,Mvir,x,y,z,rvir"
        h_dt = np.dtype([('ID', '<i8'), ('IDhost', '<i8'), ('Mvir', '<f8'), ('x', '<f8'), ('y', '<f8'),
                         ('z', '<f8'),  ('rvir', '<f8'),  ('cvir', '<f8')])
        
        #h_dt = np.dtype([('ID', '<i8'), ('IDhost', '<i8'), ('x', '<f8'), ('y', '<f8'),
        #                 ('z', '<f8'), ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'),
        #                 ('Mv', '<f8'), ('mpeak', '<f8'), ('vmp', '<f8'), ('r', '<f8'),
        #                 ('sm', '<f8'), ('icl', '<f8'), ('sfr', '<f8'), ('ssfr', '<f8'),
        #                 ('pid', '<f8'), ('Mvir', '<f8'), ('rvir', '<f8'),
        #                 ('rs_hlist', '<f8'), ('scale_half_mass', '<f8'), ('scale_last_mm', '<f8'),
        #                 ('m200b_hlist', '<f8'), ('m200c_hlist', '<f8'), ('gamma_inst', '<f8'),
        #                 ('gamma_100myr', '<f8'), ('gamma_1tdyn', '<f8'), ('gamma_2tdyn', '<f8'),
        #                 ('gamma_mpeak', '<f8'), ('vmax_mpeak', '<f8'), ('halo_hostid', '<i8'),
        #                 ('mhalo_host', '<f8'), ('mask_central', '?'), ('mtot_galaxy', '<f8'),
        #                 ('mstar_mhalo', '<f8'), ('logms_gal', '<f8'), ('logms_icl', '<f8'),
        #                 ('logms_tot', '<f8'), ('logms_halo', '<f8'), ('logmh_vir', '<f8'),
        #                 ('logmh_peak', '<f8'), ('logmh_host', '<f8'),('cvir','<f8')])

        h = np.zeros(h_header['Nsubgroups_Total'], dtype=h_dt)
        
        h['x'], h['y'], h['z'] = h_pos[:,0], h_pos[:,1], h_pos[:,2]
        h['Mvir']   = h_Mvir
        h['rvir']   = h_rvir
        h['cvir']   = h_cvir
        h['ID']     = h_ID
        h['IDhost'] = h_IDhost
        
        #select haloes above Mmin
        h = h[h['Mvir']>param.code.Mhalo_min]
        print('Nhalo = ',len(h['Mvir']))

    elif (halo_file_format=='ramses'):
        import re
        import numpy as np
        from astropy.io import ascii
        from miniramses.utils.py.miniramses import rd_halo, HaloCat, Info, rd_info

        # Extracting the path using regular expression
        path = re.match(r'(.*/miniramses/)', halo_file_in).group(1)

        # Extracting the number as an integer
        nout = int(re.match(r'.*/output_(\d+)', halo_file_in).group(1))

        i = rd_info(nout, path=path)
        cat = rd_halo(nout, path=path)
        
        unit_l = i.unit_l #cm
        unit_d = i.unit_d #g/cm^3
        unit_t = i.unit_t #s
        
        unit_v = unit_l/unit_t #cm/s
        unit_m = unit_d * (unit_l)**3.0
        
        #We convert to Mpc/h and Msol/h
        
        unit_l_in_Mpc_per_h = unit_l / (3.086e24) * param.cosmo.h0
        
        unit_m_in_Msol_per_h = unit_m / (1.989e33) * param.cosmo.h0
        
        h_dt = np.dtype([('ID', '<i8'), ('IDhost', '<i8'), ('Mvir', '<f8'), ('x', '<f8'), ('y', '<f8'),
                         ('z', '<f8'),  ('rvir', '<f8'),  ('cvir', '<f8')])
        
        h = np.zeros(len(cat.index), dtype=h_dt)
        
        h['x'], h['y'], h['z'] = cat.x*unit_l_in_Mpc_per_h, cat.y*unit_l_in_Mpc_per_h, cat.z*unit_l_in_Mpc_per_h
        h['Mvir']   = cat.mass*unit_m_in_Msol_per_h
        h['rvir']   = cat.rvir*unit_l_in_Mpc_per_h
        h['cvir']   = cat.cvir
        h['ID']     = cat.index
        h['IDhost'] = -1.0
        
        h = h[h['Mvir']>param.code.Mhalo_min]
        
    else:
        print('Unknown halo file format. Exit!')
        exit()

    #build buffer taking advantage of the periodic boundary conditions
    rbuffer = param.code.rbuffer
    ID = np.where((h['x']>(Lbox-rbuffer)) & (h['x']<=Lbox))
    h  = np.append(h,h[ID])
    if (len(ID[0])>0):
        h['x'][-len(ID[0]):] = h['x'][-len(ID[0]):]-Lbox
    ID = np.where((h['x']>0) & (h['x']<rbuffer))
    h  = np.append(h,h[ID])
    if (len(ID[0])>0):
        h['x'][-len(ID[0]):] = h['x'][-len(ID[0]):]+Lbox
    ID = np.where((h['y']>(Lbox-rbuffer)) & (h['y']<=Lbox))
    h  = np.append(h,h[ID])
    if (len(ID[0])>0):
        h['y'][-len(ID[0]):] = h['y'][-len(ID[0]):]-Lbox
    ID = np.where((h['y']>0) & (h['y']<rbuffer))
    h  = np.append(h,h[ID])
    if (len(ID[0])>0):
        h['y'][-len(ID[0]):] = h['y'][-len(ID[0]):]+Lbox
    ID = np.where((h['z']>(Lbox-rbuffer)) & (h['z']<=Lbox))
    h  = np.append(h,h[ID])
    if (len(ID[0])>0):
        h['z'][-len(ID[0]):] = h['z'][-len(ID[0]):]-Lbox
    ID = np.where((h['z']>0) & (h['z']<rbuffer))
    h  = np.append(h,h[ID])
    if (len(ID[0])>0):
        h['z'][-len(ID[0]):] = h['z'][-len(ID[0]):]+Lbox

    #separate into chunks
    N_chunk = param.sim.N_chunk
    L_chunk = Lbox/N_chunk
    h_list = []
    
    for x_min in np.linspace(0,Lbox-L_chunk,N_chunk):
        x_max = x_min + L_chunk
        for y_min in np.linspace(0,Lbox-L_chunk,N_chunk):
            y_max =y_min + L_chunk
            for z_min in np.linspace(0,Lbox-L_chunk,N_chunk):
                z_max =z_min + L_chunk
                ID = np.where((h['x']>=(x_min-rbuffer)) & (h['x']<(x_max+rbuffer)) & \
                              (h['y']>=(y_min-rbuffer)) & (h['y']<(y_max+rbuffer)) & \
                              (h['z']>=(z_min-rbuffer)) & (h['z']<(z_max+rbuffer)))
                                
                h_list += [h[ID]]
    
    return h_list


def write_halo_file(h,param):
    """
    Write halo file with displaced halo centres
    (no mass correction!)
    """
    halo_file_out = param.files.halofile_out
    halo_file_format = param.files.halofile_format
    Lbox = param.sim.Lbox

    #remove buffer
    h = h[h['x']<=Lbox]
    h = h[h['y']<=Lbox]
    h = h[h['z']<=Lbox]
    h = h[h['x']>=0]
    h = h[h['y']>=0]
    h = h[h['z']>=0]

    #only AHF_ASCII at the moment
    halo_file_format = 'AHF-ASCII'

    #Halo file format
    if (halo_file_format=='AHF-ASCII'):

        h_dt = np.dtype([('ID', '<i8'), ('IDhost', '<i8'), ('numSubStruc', '<i8'), ('Mvir', '<f8'), ('Nvir', '<f8'),
            ('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'),
            ('rvir', '<f8'), ('rmax', '<f8'), ('r2', '<f8'), ('mbp_offset', '<f8'), ('com_offset', '<f8'),
            ('v_max', '<f8'), ('v_esc', '<f8'), ('sigV', '<f8'), ('lambda', '<f8'), ('lambdaE', '<f8'),
            ('Lx', '<f8'), ('Ly', '<f8'), ('Lz', '<f8'), ('b', '<f8'),('c', '<f8'),
            ('Eax', '<f8'), ('Eay', '<f8'), ('Eaz', '<f8'), ('Ebx', '<f8'), ('Eby', '<f8'), ('Ebz', '<f8'),
            ('Ecx', '<f8'), ('Ecy', '<f8'), ('Ecz', '<f8'), ('ovdens', '<f8'), ('nbins', '<f8'),
            ('fMhires', '<f8'), ('Ekin', '<f8'), ('Epot', '<f8'), ('SurfP', '<f8'), ('Phi0', '<f8'),
            ('cvir', '<f8'), ('n_gas', '<f8'), ('M_gas', '<f8'), ('lambda_gas', '<f8'), ('lambdaE_gas', '<f8'),
            ('Lx_gas', '<f8'), ('Ly_gas', '<f8'), ('Lz_gas', '<f8'), ('b_gas', '<f8'),('c_gas', '<f8'),
            ('Eax_gas', '<f8'), ('Eay_gas', '<f8'), ('Eaz_gas', '<f8'), ('Ebx_gas', '<f8'), ('Eby_gas', '<f8'),
            ('Ebz_gas', '<f8'), ('Ecx_gas', '<f8'), ('Ecy_gas', '<f8'), ('Ecz_gas', '<f8'),
            ('Ekin_gas', '<f8'), ('Epot_gas', '<f8'), ('n_star', '<f8'), ('M_star', '<f8'),
            ('lambda_star', '<f8'), ('lambdaE_star', '<f8'),
            ('Lx_star', '<f8'), ('Ly_star', '<f8'), ('Lz_star', '<f8'), ('b_star', '<f8'),('c_star', '<f8'),
            ('Eax_star', '<f8'), ('Eay_star', '<f8'), ('Eaz_star', '<f8'), ('Ebx_star', '<f8'), ('Eby_star', '<f8'),
            ('Ebz_star', '<f8'), ('Ecx_star', '<f8'), ('Ecy_star', '<f8'), ('Ecz_star', '<f8'),
            ('Ekin_star', '<f8'), ('Epot_star', '<f8')])

        h_out = np.zeros(len(h['x']), dtype=h_dt)

        header = "ID(1) hostHalo(2) numSubStruct(3) Mvir(4) npart(5) Xc(6) Yc(7) Zc(8)\
                  VXc(9) VYc(10) VZc(11) Rvir(12) Rmax(13) r2(14)  mbp_offset(15) com_offset(16)\
                  Vmax(17) v_esc(18) sigV(19) lambda(20) lambdaE(21) Lx(22) Ly(23) Lz(24)\
                  b(25) c(26) Eax(27) Eay(28) Eaz(29) Ebx(30) Eby(31) Ebz(32) Ecx(33) Ecy(34)\
                  Ecz(35) ovdens(36) nbins(37) fMhires(38) Ekin(39) Epot(40) SurfP(41) Phi0(42)\
                  cNFW(43) n_gas(44) M_gas(45) lambda_gas(46) lambdaE_gas(47) Lx_gas(48)\
                  Ly_gas(49) Lz_gas(50) b_gas(51) c_gas(52) Eax_gas(53) Eay_gas(54) Eaz_gas(55)\
                  Ebx_gas(56) Eby_gas(57) Ebz_gas(58) Ecx_gas(59) Ecy_gas(60) Ecz_gas(61)\
                  Ekin_gas(62) Epot_gas(63) n_star(64) M_star(65) lambda_star(66)\
                  lambdaE_star(67) Lx_star(68) Ly_star(69) Lz_star(70) b_star(71) c_star(72)\
                  Eax_star(73) Eay_star(74) Eaz_star(75) Ebx_star(76) Eby_star(77) Ebz_star(78)\
                  Ecx_star(79) Ecy_star(80) Ecz_star(81) Ekin_star(82) Epot_star(83)"

        h_out['ID']     = h['ID']
        h_out['IDhost'] = h['IDhost']
        h_out['Mvir']   = h['Mvir']
        h_out['x']      = h['x']*1000.0
        h_out['y']      = h['y']*1000.0
        h_out['z']      = h['z']*1000.0
        h_out['rvir']   = h['rvir']*1000.0
        h_out['cvir']   = h['cvir']

        
        np.savetxt(halo_file_out, h_out, delimiter='\t', newline='\n', header=header, comments='# ', encoding=None)
        
    elif (halo_file_format=='ramses'):
        h_dt = np.dtype([('ID', '<i8'), ('IDhost', '<i8'), ('numSubStruc', '<i8'), ('Mvir', '<f8'), ('Nvir', '<f8'),
            ('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'),
            ('rvir', '<f8'), ('rmax', '<f8'), ('r2', '<f8'), ('mbp_offset', '<f8'), ('com_offset', '<f8'),
            ('v_max', '<f8'), ('v_esc', '<f8'), ('sigV', '<f8'), ('lambda', '<f8'), ('lambdaE', '<f8'),
            ('Lx', '<f8'), ('Ly', '<f8'), ('Lz', '<f8'), ('b', '<f8'),('c', '<f8'),
            ('Eax', '<f8'), ('Eay', '<f8'), ('Eaz', '<f8'), ('Ebx', '<f8'), ('Eby', '<f8'), ('Ebz', '<f8'),
            ('Ecx', '<f8'), ('Ecy', '<f8'), ('Ecz', '<f8'), ('ovdens', '<f8'), ('nbins', '<f8'),
            ('fMhires', '<f8'), ('Ekin', '<f8'), ('Epot', '<f8'), ('SurfP', '<f8'), ('Phi0', '<f8'),
            ('cvir', '<f8'), ('n_gas', '<f8'), ('M_gas', '<f8'), ('lambda_gas', '<f8'), ('lambdaE_gas', '<f8'),
            ('Lx_gas', '<f8'), ('Ly_gas', '<f8'), ('Lz_gas', '<f8'), ('b_gas', '<f8'),('c_gas', '<f8'),
            ('Eax_gas', '<f8'), ('Eay_gas', '<f8'), ('Eaz_gas', '<f8'), ('Ebx_gas', '<f8'), ('Eby_gas', '<f8'),
            ('Ebz_gas', '<f8'), ('Ecx_gas', '<f8'), ('Ecy_gas', '<f8'), ('Ecz_gas', '<f8'),
            ('Ekin_gas', '<f8'), ('Epot_gas', '<f8'), ('n_star', '<f8'), ('M_star', '<f8'),
            ('lambda_star', '<f8'), ('lambdaE_star', '<f8'),
            ('Lx_star', '<f8'), ('Ly_star', '<f8'), ('Lz_star', '<f8'), ('b_star', '<f8'),('c_star', '<f8'),
            ('Eax_star', '<f8'), ('Eay_star', '<f8'), ('Eaz_star', '<f8'), ('Ebx_star', '<f8'), ('Eby_star', '<f8'),
            ('Ebz_star', '<f8'), ('Ecx_star', '<f8'), ('Ecy_star', '<f8'), ('Ecz_star', '<f8'),
            ('Ekin_star', '<f8'), ('Epot_star', '<f8')])

        h_out = np.zeros(len(h['x']), dtype=h_dt)

        header = "ID(1) hostHalo(2) numSubStruct(3) Mvir(4) npart(5) Xc(6) Yc(7) Zc(8)\
                  VXc(9) VYc(10) VZc(11) Rvir(12) Rmax(13) r2(14)  mbp_offset(15) com_offset(16)\
                  Vmax(17) v_esc(18) sigV(19) lambda(20) lambdaE(21) Lx(22) Ly(23) Lz(24)\
                  b(25) c(26) Eax(27) Eay(28) Eaz(29) Ebx(30) Eby(31) Ebz(32) Ecx(33) Ecy(34)\
                  Ecz(35) ovdens(36) nbins(37) fMhires(38) Ekin(39) Epot(40) SurfP(41) Phi0(42)\
                  cNFW(43) n_gas(44) M_gas(45) lambda_gas(46) lambdaE_gas(47) Lx_gas(48)\
                  Ly_gas(49) Lz_gas(50) b_gas(51) c_gas(52) Eax_gas(53) Eay_gas(54) Eaz_gas(55)\
                  Ebx_gas(56) Eby_gas(57) Ebz_gas(58) Ecx_gas(59) Ecy_gas(60) Ecz_gas(61)\
                  Ekin_gas(62) Epot_gas(63) n_star(64) M_star(65) lambda_star(66)\
                  lambdaE_star(67) Lx_star(68) Ly_star(69) Lz_star(70) b_star(71) c_star(72)\
                  Eax_star(73) Eay_star(74) Eaz_star(75) Ebx_star(76) Eby_star(77) Ebz_star(78)\
                  Ecx_star(79) Ecy_star(80) Ecz_star(81) Ekin_star(82) Epot_star(83)"

        h_out['ID']     = h['ID']
        h_out['IDhost'] = h['IDhost']
        h_out['Mvir']   = h['Mvir']
        h_out['x']      = h['x']*1000.0
        h_out['y']      = h['y']*1000.0
        h_out['z']      = h['z']*1000.0
        h_out['rvir']   = h['rvir']*1000.0
        h_out['cvir']   = h['cvir']

        
        np.savetxt(halo_file_out, h_out, delimiter='\t', newline='\n', header=header, comments='# ', encoding=None)


    else:
        print("Try: halo_file_format==AHF-ASCII. No other formats implemented")
        exit()

    return
