import psi4
import mdtraj as md
import numpy as np

def trajectory_from_coordinates(mol, geometries):
    natom = mol.natom()
    bonds = psi4.qcdb.parker._bond_profile(mol)
    n_frames = len(geometries)
    
    xyz = np.zeros((n_frames, natom, 3))
    for frame in range(n_frames):
        xyz[frame, :, :] = np.array(geometries[frame]) * psi4.constants.bohr2angstroms / 10.0 # in nm
    
    top = md.Topology()
    chain = top.add_chain()
    res = top.add_residue("RES", chain)
    for i in range(natom):
        element = md.element.Element.getBySymbol(mol.symbol(i))
        top.add_atom(mol.symbol(i), element, res)
    for bond in bonds:
        top.add_bond(top.atom(bond[0]), top.atom(bond[1]), order=bond[2])
    traj = md.Trajectory(xyz, top)
    
    return traj

def trajectory_from_xyz(mol, file_name):
    natom = mol.natom()
    bonds = psi4.qcdb.parker._bond_profile(mol)
    
    top = md.Topology()
    chain = top.add_chain()
    res = top.add_residue("RES", chain)
    for i in range(natom):
        element = md.element.Element.getBySymbol(mol.symbol(i))
        top.add_atom(mol.symbol(i), element, res)
    for bond in bonds:
        top.add_bond(top.atom(bond[0]), top.atom(bond[1]), order=bond[2])
        
    traj = md.load(file_name, top=top)
    
    return traj

def vibrational_modes_trajectories(wfn):
    import mdtraj as md
    import numpy as np

    mol = wfn.molecule()
    natom = mol.natom()
    vibinfo = wfn.frequency_analysis
    
    vib_mode = [idx for idx, trv in enumerate(vibinfo['TRV'].data) if trv == 'V']
    normal_modes = {}

    for mode in vib_mode:
        freq = float(vibinfo['omega'].data[mode].real)
        normal_modes[freq] = {}
        displacement = vibinfo['x'].data[:, mode].reshape(natom, 3)
        normal_modes[freq]['displaced_geometry'] = mol.geometry().np + displacement
        try:
            normal_modes[freq]['intensity'] = float(vibinfo['IR_intensity'].data[mode].real)
        except:
            print("Intensity not available; setting to 1")
            normal_modes[freq]['intensity'] = 1.0
        
    bonds = psi4.qcdb.parker._bond_profile(mol)
    
    trajectories = []
    for freq in normal_modes:
        xyz = np.zeros((2, natom, 3))
        xyz[0, :, :] = mol.geometry().np
        xyz[1, :, :] = normal_modes[freq]['displaced_geometry']
        xyz *= psi4.constants.bohr2angstroms / 10.0 # in nm
    
        top = md.Topology()
        chain = top.add_chain()
        res = top.add_residue("RES", chain)
        for i in range(natom):
            element = md.element.Element.getBySymbol(mol.symbol(i))
            top.add_atom(mol.symbol(i), element, res)
        for bond in bonds:
            top.add_bond(top.atom(bond[0]), top.atom(bond[1]), order=bond[2])
        traj = md.Trajectory(xyz, top)
        trajectories.append(traj)
        
    # Broaden the peaks
    # https://github.com/fevangelista/Course-QuantumChemistryLab/blob/master/Notebooks/03-StationaryPoints/03-StationaryPoints.ipynb
    import math
    frequencies = []
    intensities = []
    for freq in normal_modes.keys():
        frequencies.append(freq)
        intensities.append(normal_modes[freq]['intensity'])
    
    xmin = 0
    xmax = 5000
    npoints = 600
    alpha = 1.0 / 100.0
    dx = (xmax - xmin)/ float(npoints)
    xvals = [xmin + dx * i for i in range(npoints)]
    yvals = []
    for x in xvals:
        y = 0.0
        for f,i in zip(frequencies,intensities):
            y += i * math.exp(- alpha * (x - f)**2)
        yvals.append(y)

    
    return trajectories, xvals, yvals