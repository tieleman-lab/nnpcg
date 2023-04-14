# runner module in the nnpcg package
# author: Daniel P.Ramirez
import torch
import numpy as np
from tqdm import tqdm
from htmd.ui import *
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


class EasySimulator:
    def __init__(
        self,
        topology="structure.prmtop",
        coordinates="input.coor",
        box_dimensions="input.xsc",
        precision=torch.float,
        device="cpu",
    ):
        # Initialize the class with given inputs
        self.topology = topology
        self.mol = Molecule(self.topology)  # Reading the system topology
        self.mol.read(coordinates)  # Reading the initial simulation coordinates
        self.mol.read(box_dimensions)  # Reading the box dimensions
        self.precision = precision  # Set the precision for calculations
        self.device = device  # Set the device for computations
        self.system = None  # Initialize the system object
        self.forces = None  # Initialize the forces object
        self.parameters = None  # Initialize the parameters object
        self.integrator = None  # Initialize the integrator object
        self.wrapper = None  # Initialize the wrapper object

    def build_system(self, temperature=300):
        # Create a forcefield object and set system parameters
        ff = ForceField.create(self.mol, self.topology)
        self.parameters = Parameters(
            ff, self.mol, precision=self.precision, device=self.device
        )
        self.system = System(
            self.mol.numAtoms, nreplicas=1, precision=self.precision, device=self.device
        )
        self.system.set_positions(
            self.mol.coords
        )  # Set the atomic positions for the system
        self.system.set_box(self.mol.box)  # Set the box dimensions for the system
        self.system.set_velocities(
            maxwell_boltzmann(self.parameters.masses, T=temperature, replicas=1)
        )  # Set velocities for the system using Maxwell-Boltzmann distribution

    def minimize_system(
        self, langevin_temperature=300, langevin_gamma=0.1, timestep=1, steps=500
    ):
        # Create forces object, integrator object, and wrapper object
        self.forces = Forces(
            self.parameters,
            cutoff=9,
            rfa=True,
            switch_dist=7.5,
            terms=[
                "bonds",
                "angles",
                "dihedrals",
                "impropers",
                "1-4",
                "electrostatics",
                "lj",
            ],
        )
        self.integrator = Integrator(
            self.system,
            self.forces,
            timestep,
            self.device,
            gamma=langevin_gamma,
            T=langevin_temperature,
        )
        self.wrapper = Wrapper(
            self.mol.numAtoms,
            self.mol.bonds if len(self.mol.bonds) else None,
            self.device,
        )
        minimize_bfgs(self.system, self.forces, steps=500)  # Minimize the system

    def run_simulation(
        self,
        logger_path="logs/",
        logger_name="monitor.csv",
        timestep=1,
        steps=1000,
        output_period=10,
        save_period=100,
        trajectoryout="trajectory.npy",
    ):
        # Initialize a logger object and a trajectory array
        logger = LogWriter(
            path=logger_path,
            keys=("iter", "ns", "epot", "ekin", "etot", "T"),
            name=logger_name,
        )
        FS2NS = 1e-6  # Femtosecond to nanosecond conversion
        traj = []

        # Use tqdm to display progress bar while iterating over simulation steps
        iterator = tqdm(range(1, int(steps / output_period) + 1))
        Epot = self.forces.compute(
            self.system.pos, self.system.box, self.system.forces
        )  # Calculate the potential energy of the system
        for i in iterator:
            Ekin, Epot, T = self.integrator.step(
                niter=output_period
            )  # Perform simulation step and calculate energy and temperature
            self.wrapper.wrap(self.system.pos, self.system.box)
            currpos = self.system.pos.detach().cpu().numpy().copy()
            traj.append(currpos)

            if (i * output_period) % save_period == 0:
                np.save(trajectoryout, np.stack(traj, axis=2))

            logger.write_row(
                {
                    "iter": i * output_period,
                    "ns": FS2NS * i * output_period * timestep,
                    "epot": Epot,
                    "ekin": Ekin,
                    "etot": Epot + Ekin,
                    "T": T,
                }
            )

    # Print a command for the user to run in a terminal with the proper python environment
    def run_from_config(self):
        print(
            "Please run the following command in a terminal with the proper python env:"
        )
        print("python train.py --conf simulate.yaml --log-dir data/train_light")
