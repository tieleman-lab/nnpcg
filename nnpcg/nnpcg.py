# main module in the nnpcg package
# author: Daniel P.Ramirez
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm 
from moleculekit.molecule import Molecule
from torchmd.forcefields.forcefield import ForceField
from torchmd.parameters import Parameters
from torchmd.integrator import maxwell_boltzmann
from torchmd.systems import System
from torchmd.forces import Forces
from torchmd.integrator import Integrator
from torchmd.wrapper import Wrapper
from torchmd.minimizers import minimize_bfgs
from torchmd.utils import LogWriter
from torchmd_cg.utils.psfwriter import pdb2psf_CA
from torchmd_cg.utils.prior_fit import get_param_bonded
from torchmd_cg.utils.prior_fit import get_param_nonbonded
from torchmd_cg.utils.make_deltaforces import make_deltaforces


class EasyTrainer():
    def __init__(self, pdb_file='coordinates.pdb', psf_file='topology.psf', coords_array='coods_array.npy'):
        self.structure = pdb_file
        self.topology = psf_file
        self.coords_array_file = coords_array
        pdb2psf_CA(self.structure, self.topology, bonds = True, angles = False)
        self.mol = Molecule(psf_file)  
        self.coords_array = np.load(self.coords_array_file)
        self.mol.coords = np.moveaxis(self.coords_array, 0, -1) 
        self.priors = {}
        self.priors['atomtypes'] = list(set(self.mol.atomtype))
        self.priors['bonds'] = {}
        self.priors['lj'] = {}
        self.priors['electrostatics'] = {at: {'charge': 0.0} for at in self.priors['atomtypes']}
        self.priors['masses'] = {at: 12.0 for at in self.priors['atomtypes']}

    def fit_bonded_params(self, temperature=300, fit_range=[3.55, 4.2]):
        self.priors['bonds'] = get_param_bonded(self.mol, fit_range, temperature)

    def fit_nonbonded_params(self, temperature=300, fit_range=[3, 6]):
        self.priors['lj'] = get_param_nonbonded(self.mol, fit_range, temperature)

    def write_forcefield(self, ff_file='forcefield.yaml'):
        with open(ff_file, "w") as f: 
            yaml.dump(self.priors, f)

    def calculate_deltaforces(self, forces='forces.npy', forcefield='forcefield.yaml', deltaforces='deltaforces.npy', device='cpu'):
        exclusions = ('bonds')
        forceterms = ['Bonds', 'RepulsionCG']
        make_deltaforces(self.coords_array_file, forces, deltaforces, forcefield, self.topology, exclusions, device, forceterms)
    
    def generate_embeddings(self, embedding_dict, embedding_file='embeddings.npy'):
        emb = np.array([embedding_dict[x] for x in self.mol.resname], dtype='<U3')
        np.save(embedding_file, emb)

    def train_command(self):
        print('Please run the following command in a terminal with the proper python env:')
        print('python train.py --conf train.yaml --log-dir data/train_light')


class EasySimulator():
    def __init__(self, topology='structure.prmtop', coordinates='input.coor', box_dimensions='input.xsc', precision=torch.float, device='cpu'):
        self.topology = topology
        self.mol = Molecule(self.topology)  # Reading the system topology
        self.mol.read(coordinates)  # Reading the initial simulation coordinates
        self.mol.read(box_dimensions)  # Reading the box dimensions
        self.precision = precision
        self.device = device
        self.system = None
        self.forces = None
        self.parameters = None
        self.integrator = None
        self.wrapper = None

    def build_system(self, temperature=300):
        ff = ForceField.create(self.mol, self.topology)
        self.parameters = Parameters(ff, self.mol, precision=self.precision, device=self.device)
        self.system = System(self.mol.numAtoms, nreplicas=1, precision=self.precision, device=self.device)
        self.system.set_positions(self.mol.coords)
        self.system.set_box(self.mol.box)
        self.system.set_velocities(maxwell_boltzmann(self.parameters.masses, T=temperature, replicas=1))

    def minimize_system(self, langevin_temperature=300, langevin_gamma=0.1, timestep=1, steps=500):
        self.forces = Forces(self.parameters, cutoff=9, rfa=True, switch_dist=7.5, terms=["bonds", "angles", "dihedrals", "impropers", "1-4", "electrostatics", "lj"])

        self.integrator = Integrator(self.system, self.forces, timestep, self.device, gamma=langevin_gamma, T=langevin_temperature)
        self.wrapper = Wrapper(self.mol.numAtoms, self.mol.bonds if len(self.mol.bonds) else None, self.device)

        minimize_bfgs(self.system, self.forces, steps=500)  # Minimize the system

    def run_simulation(self, logger_path="logs/", logger_name="monitor.csv", timestep=1, steps=1000, output_period=10, save_period=100, trajectoryout="trajectory.npy"):
        logger = LogWriter(path=logger_path, keys=('iter','ns','epot','ekin','etot','T'), name=logger_name)
        FS2NS = 1E-6 # Femtosecond to nanosecond conversion
        traj = []

        iterator = tqdm(range(1, int(steps / output_period) + 1))
        Epot = self.forces.compute(self.system.pos, self.system.box, self.system.forces)
        for i in iterator:
            Ekin, Epot, T = self.integrator.step(niter=output_period)
            self.wrapper.wrap(self.system.pos, self.system.box)
            currpos = self.system.pos.detach().cpu().numpy().copy()
            traj.append(currpos)
            
            if (i*output_period) % save_period  == 0:
                np.save(trajectoryout, np.stack(traj, axis=2))

            logger.write_row({'iter':i*output_period,'ns':FS2NS*i*output_period*timestep,'epot':Epot,'ekin':Ekin,'etot':Epot+Ekin,'T':T})
    
    def run_from_config(self):
        print('Please run the following command in a terminal with the proper python env:')
        print('python train.py --conf simulate.yaml --log-dir data/train_light')


# def EasyAnalyzer():
