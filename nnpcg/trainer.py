# trainer module in the nnpcg package
# author: Daniel P.Ramirez
import yaml
import numpy as np
from htmd.ui import *
from moleculekit.molecule import Molecule
from torchmd_cg.utils.psfwriter import pdb2psf_CA
from torchmd_cg.utils.prior_fit import get_param_bonded
from torchmd_cg.utils.prior_fit import get_param_nonbonded
from torchmd_cg.utils.make_deltaforces import make_deltaforces


class EasyTrainer:
    def __init__(
        self,
        pdb_file="coordinates.pdb",
        psf_file="topology.psf",
        coords_array="coods_array.npy",
    ):
        # Initialize the class with default file names or user-provided names
        self.structure = pdb_file
        self.topology = psf_file
        self.coords_array_file = coords_array
        # Convert PDB coordinates to PSF format and read the structure as a Molecule object
        pdb2psf_CA(self.structure, self.topology, bonds=True, angles=False)
        self.mol = Molecule(psf_file)
        # Load pre-calculated coordinate array and assign it to the Molecule object
        self.coords_array = np.load(self.coords_array_file)
        self.mol.coords = np.moveaxis(self.coords_array, 0, -1)
        # Set up the default priors for atomtypes, bonds, LJ, electrostatics, and masses
        self.priors = {}
        self.priors["atomtypes"] = list(set(self.mol.atomtype))
        self.priors["bonds"] = {}
        self.priors["lj"] = {}
        self.priors["electrostatics"] = {
            at: {"charge": 0.0} for at in self.priors["atomtypes"]
        }
        self.priors["masses"] = {at: 12.0 for at in self.priors["atomtypes"]}

    def fit_bonded_params(self, temperature=300, fit_range=[3.55, 4.2]):
        # Fit bonded parameters (bonds) using the provided temperature and fit range
        self.priors["bonds"] = get_param_bonded(self.mol, fit_range, temperature)

    def fit_nonbonded_params(self, temperature=300, fit_range=[3, 6]):
        # Fit non-bonded parameters (LJ) using the provided temperature and fit range
        self.priors["lj"] = get_param_nonbonded(self.mol, fit_range, temperature)

    def write_forcefield(self, ff_file="forcefield.yaml"):
        # Write the forcefield YAML file containing the priors
        with open(ff_file, "w") as f:
            yaml.dump(self.priors, f)

    def calculate_deltaforces(
        self,
        forces="forces.npy",
        forcefield="forcefield.yaml",
        deltaforces="deltaforces.npy",
        device="cpu",
    ):
        # Calculate the difference between the forcefield and the original force array, excluding the bonded terms
        exclusions = "bonds"
        forceterms = ["Bonds", "RepulsionCG"]
        make_deltaforces(
            self.coords_array_file,
            forces,
            deltaforces,
            forcefield,
            self.topology,
            exclusions,
            device,
            forceterms,
        )

    def generate_embeddings(self, embedding_dict, embedding_file="embeddings.npy"):
        # Generate embeddings based on residue names and save them to a numpy file
        emb = np.array([embedding_dict[x] for x in self.mol.resname], dtype="<U3")
        np.save(embedding_file, emb)

    def train_command(self):
        # Print a command for the user to run in a terminal with the proper python environment
        print(
            "Please run the following command in a terminal with the proper python env:"
        )
        print(
            "python nnpcg/train.py --conf nnpcg/train.yaml --log-dir nnpcg/data/train_light"
        )
