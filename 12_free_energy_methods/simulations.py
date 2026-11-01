"""Simulation functions used in the free energy notes.

The results of these simulations are provided in the data folder. The functions are kept
here so that the notebook stays readable and so that the data can be regenerated.
"""
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

# ------------------------------------------------------------------ hydration free energies

def build_solvated_system(smiles, box_length=2.4):
    """Parametrize a small molecule with OpenFF Sage and put it in a cubic box of TIP3P water."""
    from openff.toolkit import Molecule, ForceField, Topology
    from openff.interchange import Interchange
    from openff.units import unit as off_unit

    # The solute: from SMILES to a 3D structure with OpenFF (NAGL AM1-BCC) charges
    solute = Molecule.from_smiles(smiles)
    solute.generate_conformers(n_conformers=1)
    solute_positions = solute.conformers[0].m_as("nanometer")
    solute_positions -= solute_positions.mean(axis=0) - box_length / 2  # move to the box center

    # A pre-equilibrated water box from OpenMM, with the waters that overlap the solute removed
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(app.ForceField("tip3p.xml"), boxSize=mm.Vec3(box_length, box_length, box_length) * unit.nanometer)
    water_xyz = modeller.getPositions().value_in_unit(unit.nanometer)
    water_xyz = np.array([[v.x, v.y, v.z] for v in water_xyz])
    keep = []
    for i, residue in enumerate(modeller.topology.residues()):
        d = water_xyz[3 * i] - solute_positions
        d -= box_length * np.round(d / box_length)  # minimum image
        if np.min(np.linalg.norm(d, axis=1)) > 0.3:
            keep.append(i)
    n_waters = len(keep)
    water_xyz = np.concatenate([water_xyz[3 * i:3 * i + 3] for i in keep])

    # One OpenFF topology for the whole box; Sage contains the TIP3P parameters for water
    water = Molecule.from_smiles("O")
    topology = Topology.from_molecules([solute] + [water] * n_waters)
    topology.box_vectors = np.eye(3) * box_length * off_unit.nanometer
    positions = np.vstack([solute_positions, water_xyz])
    topology.set_positions(positions * off_unit.nanometer)

    interchange = Interchange.from_smirnoff(ForceField("openff-2.3.0.offxml"), topology)
    return solute, interchange.to_openmm_topology(), interchange.to_openmm_system(), positions, n_waters


def make_alchemical(system, solute_atoms):
    """Decouple the solute from its environment with two lambda parameters.
    lambda_elec scales the solute charges (solute-solvent electrostatics),
    lambda_vdw turns off solute-solvent Lennard-Jones with a soft-core potential.
    All solute-solute interactions are kept as exceptions so they do not change with lambda."""
    nonbonded = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
    solute_atoms = set(solute_atoms)
    solvent_atoms = set(range(system.getNumParticles())) - solute_atoms

    # Keep every intramolecular solute pair as an exception with its full interaction
    existing = set()
    for k in range(nonbonded.getNumExceptions()):
        i, j, *_ = nonbonded.getExceptionParameters(k)
        existing.add((min(i, j), max(i, j)))
    for i in solute_atoms:
        for j in solute_atoms:
            if i < j and (i, j) not in existing:
                qi, si, ei = nonbonded.getParticleParameters(i)
                qj, sj, ej = nonbonded.getParticleParameters(j)
                nonbonded.addException(i, j, qi * qj, (si + sj) / 2, (ei * ej).sqrt())

    # Soft-core Lennard-Jones between solute and solvent
    energy = ("lambda_vdw*4*epsilon*x*(x-1.0); x = 1/(0.5*(1.0-lambda_vdw) + (r/sigma)^6);"
              "sigma = 0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2)")
    softcore = mm.CustomNonbondedForce(energy)
    softcore.addGlobalParameter("lambda_vdw", 1.0)
    softcore.addPerParticleParameter("sigma")
    softcore.addPerParticleParameter("epsilon")
    softcore.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    softcore.setCutoffDistance(nonbonded.getCutoffDistance())
    softcore.setUseSwitchingFunction(nonbonded.getUseSwitchingFunction())
    softcore.setSwitchingDistance(nonbonded.getSwitchingDistance())
    softcore.setUseLongRangeCorrection(nonbonded.getUseDispersionCorrection())

    nonbonded.addGlobalParameter("lambda_elec", 1.0)
    for i in range(system.getNumParticles()):
        q, sigma, epsilon = nonbonded.getParticleParameters(i)
        softcore.addParticle([sigma, epsilon])
        if i in solute_atoms:
            # charge = 0 + lambda_elec * q ;  LJ moves to the soft-core force
            nonbonded.setParticleParameters(i, 0.0, sigma, 0.0)
            nonbonded.addParticleParameterOffset("lambda_elec", i, q, 0.0, 0.0)
    for k in range(nonbonded.getNumExceptions()):
        i, j, *_ = nonbonded.getExceptionParameters(k)
        softcore.addExclusion(i, j)
    softcore.addInteractionGroup(solute_atoms, solvent_atoms)
    system.addForce(softcore)
    return system


# The alchemical path: first remove the charges (lambda_elec 1 -> 0) and then the
# Lennard-Jones interactions (lambda_vdw 1 -> 0)
lambda_elec = [1.00, 0.75, 0.50, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
lambda_vdw = [1.00, 1.00, 1.00, 1.00, 1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00]
n_states = len(lambda_elec)


def sample_state(system, topology, positions, state, n_samples=250, steps_per_sample=500,
                 equilibration_steps=25000, temperature=298.15, platform=None):
    """Run NPT dynamics at one alchemical state and evaluate the reduced potential
    of every sample at all the alchemical states (needed for MBAR). Also returns dU/dlambda."""
    system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))  # work on a copy
    system.addForce(mm.MonteCarloBarostat(1 * unit.bar, temperature * unit.kelvin))
    integrator = mm.LangevinMiddleIntegrator(temperature * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    simulation.context.setParameter("lambda_elec", lambda_elec[state])
    simulation.context.setParameter("lambda_vdw", lambda_vdw[state])
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)
    simulation.step(equilibration_steps)

    kT = (unit.MOLAR_GAS_CONSTANT_R * temperature * unit.kelvin).value_in_unit(unit.kilojoule_per_mole)
    u_kn = np.zeros((n_states, n_samples))  # reduced potential of each sample in every state
    dudl = np.zeros((n_samples, 2))         # dU/dlambda_elec and dU/dlambda_vdw (kJ/mol)

    def energy(le, lv):
        simulation.context.setParameter("lambda_elec", le)
        simulation.context.setParameter("lambda_vdw", lv)
        return simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    h = 1e-3
    for n in range(n_samples):
        simulation.step(steps_per_sample)
        for k in range(n_states):
            u_kn[k, n] = energy(lambda_elec[k], lambda_vdw[k]) / kT
        le, lv = lambda_elec[state], lambda_vdw[state]
        # central finite differences (one-sided at the end points)
        dudl[n, 0] = (energy(min(le + h, 1), lv) - energy(max(le - h, 0), lv)) / (min(le + h, 1) - max(le - h, 0))
        dudl[n, 1] = (energy(le, min(lv + h, 1)) - energy(le, max(lv - h, 0))) / (min(lv + h, 1) - max(lv - h, 0))
        # go back to the sampled state
        simulation.context.setParameter("lambda_elec", le)
        simulation.context.setParameter("lambda_vdw", lv)
    return u_kn, dudl


def compute_hydration_data(molecules, filename):
    """Run all the alchemical states for each molecule and save the results."""
    results = {}
    for name, smiles in molecules.items():
        solute, topology, system, positions, n_waters = build_solvated_system(smiles)
        system = make_alchemical(system, range(solute.n_atoms))
        u_kn, dudl = [], []
        for state in range(n_states):
            u, d = sample_state(system, topology, positions * unit.nanometer, state)
            u_kn.append(u)
            dudl.append(d)
        results[f"{name}_u_kn"] = np.concatenate(u_kn, axis=1)
        results[f"{name}_dudl"] = np.array(dudl)
    np.savez_compressed(filename, **results)


# ------------------------------------------------------------------ alanine dipeptide (GBn2)

# The atoms that define the backbone dihedrals of alanine dipeptide (0-based indices)
phi_atoms = [4, 6, 8, 14]   # C(ACE)-N-CA-C
psi_atoms = [6, 8, 14, 16]  # N-CA-C-N(NME)


def _alanine_dipeptide_system():
    pdb = app.PDBFile("data/alanine_dipeptide.pdb")
    forcefield = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds,
                                     hydrogenMass=3 * unit.amu)
    return pdb, system


def dihedral(positions, atoms):
    """Dihedral angle (radians) between four atoms."""
    p = [positions[i].value_in_unit(unit.nanometer) for i in atoms]
    b0, b1, b2 = p[0] - p[1], p[2] - p[1], p[3] - p[2]
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))


def run_umbrella_window(phi0, k=200.0, n_samples=1000, steps_per_sample=250, platform=None):
    """Simulate alanine dipeptide with a harmonic bias 0.5*k*(phi - phi0)^2 (k in kJ/mol/rad^2).
    Returns the phi and psi values (radians) sampled every 1 ps."""
    pdb, system = _alanine_dipeptide_system()
    # The umbrella potential; dphi accounts for the periodicity of the dihedral angle
    bias = mm.CustomTorsionForce("0.5*k*dphi^2; dphi = min(d, 2*pi - d); d = abs(theta - phi0); pi = 3.141592653589793")
    bias.addGlobalParameter("k", k)
    bias.addGlobalParameter("phi0", phi0)
    bias.addTorsion(*phi_atoms, [])
    system.addForce(bias)

    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtoseconds)
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
    simulation.step(12500)  # 50 ps of equilibration in the window

    samples = np.zeros((n_samples, 2))
    for i in range(n_samples):
        simulation.step(steps_per_sample)
        positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        samples[i] = [dihedral(positions, phi_atoms), dihedral(positions, psi_atoms)]
    return samples


def run_metadynamics(simulation_time=50 * unit.nanoseconds, platform=None):
    """Well-tempered metadynamics of alanine dipeptide along phi and psi.
    Returns the CV values every 1 ps and the free energy surface at 50 times during the run."""
    pdb, system = _alanine_dipeptide_system()
    variables = []
    for atoms in (phi_atoms, psi_atoms):
        cv = mm.CustomTorsionForce("theta")
        cv.addTorsion(*atoms, [])
        variables.append(app.BiasVariable(cv, -np.pi, np.pi, biasWidth=0.35, periodic=True, gridWidth=72))

    temperature = 300 * unit.kelvin
    # Gaussians of height 1 kJ/mol every 1 ps; the bias factor sets how fast the hills shrink
    meta = app.Metadynamics(system, variables, temperature, biasFactor=6, height=1.0 * unit.kilojoule_per_mole,
                            frequency=250)
    integrator = mm.LangevinMiddleIntegrator(temperature, 1 / unit.picosecond, 4 * unit.femtoseconds)
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature)

    n_chunks = 50
    steps_per_chunk = int(simulation_time / integrator.getStepSize()) // n_chunks
    cv_values, free_energies = [], []
    for chunk in range(n_chunks):
        for _ in range(steps_per_chunk // 250):
            meta.step(simulation, 250)
            cv_values.append(meta.getCollectiveVariables(simulation))
        free_energies.append(meta.getFreeEnergy().value_in_unit(unit.kilojoule_per_mole))
    return np.array(cv_values), np.array(free_energies)
