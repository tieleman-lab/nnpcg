# analyzer module in the nnpcg package
# author: Daniel P.Ramirez
import numpy as np
from htmd.ui import *
import matplotlib.pyplot as plt
from moleculekit.molecule import Molecule
from pyemma.coordinates.transform.tica import TICA as TICApyemma


def computeWeights(model):
    # Concatenate the model's data states into a single array
    stateconcat = np.concatenate(model.data.St)
    # Obtain the equilibrium distribution of frames using the model's microstate cluster and stationary distribution
    eqdist_offrame = model.msm.stationary_distribution[
        model.micro_ofcluster[stateconcat]
    ]
    # Calculate the stationary probability of each microstate
    statprob = 1 / model.data.N
    # Calculate the frame weights as the product of equilibrium distribution and stationary probability
    frameweights = eqdist_offrame * statprob[stateconcat]
    # Return the computed frame weights
    return frameweights


def sample_state(model, state, statetype, frames=50, init_frames=0.1):
    # Check whether the state type is macro
    if statetype == "macro":
        mode = "weighted"
    else:
        # If not macro, set the mode to random
        mode = "random"

    # Sample states from the model
    molsampl = model.sampleStates(
        states=[
            state,
        ],
        statetype=statetype,
        samplemode=mode,
        frames=frames,
    )[1][0]

    # If init_frames is not None, add frames to the sampled molecule
    if init_frames is not None:
        framediff = int(model.data.simlist[0].numframes[0] * init_frames)
        molsampl[:, 1] += framediff

    # Create a Molecule object using the molfile from the model data
    mol = Molecule(model.data.simlist[0].molfile)

    # Read the trajectory data for each reference in molsampl
    trajlist = []
    for ref in molsampl:
        simid = model.data.simlist[ref[0]]
        trajlist.append(simid.trajectory[0])

    # Read the trajectory data into the Molecule object
    mol.read(trajlist, frames=molsampl[:, 1])

    # Return the Molecule object and the molsampl array
    return mol, molsampl


def build_backbone(cg_mol, ref_mol):
    # this will only work reasonably for structures close to reference file

    # create a copy of the reference molecule and filter it to only include
    # the backbone atoms of protein residues (C, CA, N, O)
    mol_ref = ref_mol.copy()
    mol_ref.filter("name C CA N O and protein")

    # create an empty list to hold the backbone coordinates for each frame
    traj_coords = []

    # loop over each frame in the coarse-grained (CG) molecule
    for frame in range(cg_mol.numFrames):
        # create a copy of the CG molecule and drop all frames except the current one
        mol = cg_mol.copy()
        mol.dropFrames(keep=frame)

        # align the CG molecule to the reference molecule using only the CA atoms,
        # which will ensure that the backbone atoms are aligned
        mol.align("name CA", refmol=mol_ref, refsel="name CA")

        # create NumPy arrays of the residue IDs for the reference molecule and the CG molecule
        resid_ref = np.array(list(set(mol_ref.resid)))
        resid_mol = mol.resid

        # extract the backbone coordinates for each residue, using the reference
        # molecule as a guide for the alignment
        coords = []
        atom_ref = 0
        atom_mol = 0
        for n in range(len(resid_mol) - 2):
            if n == 0:
                # initial alignment of the first three residues
                mol_ref.align(
                    f"resid {resid_ref[n]} {resid_ref[n+1]} {resid_ref[n+2]} and name CA",
                    refmol=mol,
                    refsel=f"resid {resid_mol[n]} {resid_mol[n+1]} {resid_mol[n+2]}",
                )
                for x in range(4):
                    # if the current atom in the reference molecule is a CA,
                    # use the corresponding coordinate from the CG molecule
                    if mol_ref.name[atom_ref] == "CA":
                        coords.append(mol.coords[atom_mol])
                        atom_mol += 1
                    # otherwise, use the corresponding coordinate from the reference molecule
                    else:
                        coords.append(mol_ref.coords[atom_ref])
                    atom_ref += 1

            for x in range(4):
                if mol_ref.name[atom_ref] == "N":
                    coords.append(mol_ref.coords[atom_ref])
                elif mol_ref.name[atom_ref] == "CA":
                    # align the current residue in the reference and coarse-grained molecules
                    mol_ref.align(
                        f"resid {resid_ref[n]} {resid_ref[n+1]} {resid_ref[n+2]} and name CA",
                        refmol=mol,
                        refsel=f"resid {resid_mol[n]} {resid_mol[n+1]} {resid_mol[n+2]}",
                    )
                    # use the coordinates of the C-alpha atom from the coarse-grained molecule
                    coords.append(mol.coords[atom_mol])
                    atom_mol += 1
                else:
                    # use the coordinates of the remaining atoms from the reference molecule
                    coords.append(mol_ref.coords[atom_ref])
                atom_ref += 1

            if n == len(resid_mol) - 3:
                for x in range(4):
                    if mol_ref.name[atom_ref] == "CA":
                        # if the reference atom is a CA, use the corresponding atom from the CG molecule
                        coords.append(mol.coords[atom_mol])
                        atom_mol += 1
                    else:
                        # if the reference atom is not a CA, use the corresponding atom from the reference molecule
                        coords.append(mol_ref.coords[atom_ref])
                    atom_ref += 1

        # add the coordinates for the current frame to the trajectory
        traj_coords.append(np.stack(coords))

    # concatenate the coordinates for all frames into a single array
    mol_ref.coords = np.concatenate(traj_coords, axis=2)

    # return the reference molecule with the new coordinates
    return mol_ref


def plotstates(model, states=None, dimx=0, dimy=1, cmap="Set1", zorder=5):
    # import the matplotlib library as mpl
    import matplotlib as mpl

    # if states is None, set it to range of model.macronum
    if not states:
        states = range(model.macronum)

    # get a color map using the specified cmap and number of macros in the model
    cmap = mpl.cm.get_cmap(cmap, model.macronum)

    # iterate over each macro in the states list
    for macro in states:
        # get the centers of each cluster belonging to the current macro
        macrocenters = model.data.Centers[
            np.where(model.macro_ofcluster == macro)[0], :
        ]

        # calculate the percentage of the population belonging to the current macro
        macro_pop = round(model.eqDistribution(plot=False)[macro] * 100, 1)

        # create a scatter plot of the macro centers, using a different color for each macro
        # the color of each point is determined by its macro value
        # set the z-order of the points and add a label indicating the macro number and percentage of the population it represents
        plt.scatter(
            macrocenters[:, dimx],
            macrocenters[:, dimy],
            c=[cmap(macro) for i in range(len(macrocenters[:, 0]))],
            zorder=zorder,
            label="Macro {}-{}%".format(macro, macro_pop),
        )


def plotContour(
    data,
    weights,
    levels,
    contour=True,
    fill=True,
    dimx=0,
    dimy=1,
    bins=80,
    pad=0.5,
    cmap="Greys",
    zorder=1,
    colors="black",
    alpha=1,
):
    # Importing required libraries
    from htmd.kinetics import Kinetics

    # Finding the minimum and maximum values for the specified dimensions
    xmin, xmax = [np.min(data[:, dimx]), np.max(data[:, dimx])]
    ymin, ymax = [np.min(data[:, dimy]), np.max(data[:, dimy])]
    # Defining the histogram range based on the minimum and maximum values
    hist_range = np.array([[xmin - pad, xmax + pad], [ymin - pad, ymax + pad]])
    # Creating a 2D histogram with the specified bin size, range, and weights
    counts, xbins, ybins = np.histogram2d(
        data[:, dimx], data[:, dimy], bins=bins, range=hist_range, weights=weights
    )
    counts = counts.T
    # Calculating the energy from the histogram counts using the Kinetics class
    energy = np.where(counts != 0, -Kinetics._kB * 350 * np.log(counts), counts)
    # Setting the minimum energy to the next highest value after the minimum non-zero energy
    energy[energy == 0] = np.max(energy) + 10
    ecorr = np.min(energy[energy != 0])
    energy = np.where(energy != 0, energy - ecorr, energy)  # Setting the minimum as 0
    # Finding the centers of the bins to create a mesh grid for plotting
    xcenters = (xbins[:-1] + xbins[1:]) / 2
    ycenters = (ybins[:-1] + ybins[1:]) / 2
    meshx, meshy = np.meshgrid(xcenters, ycenters)
    # Getting the colormap and normalization function for the specified levels
    cmap, norm = getcmap(cmap, levels)
    # Plotting the contour lines if specified
    if contour:
        plt.contour(
            meshx,
            meshy,
            energy,
            levels=levels,
            colors=colors,
            vmin=0,
            vmax=levels[-1],
            linewidths=2,
            alpha=alpha,
        )
    # Plotting the filled contours if specified
    if fill:
        plt.contourf(
            meshx,
            meshy,
            energy,
            levels=levels,
            cmap=cmap,
            vmin=0,
            vmax=levels[-1],
            norm=norm,
        )


def getcmap(cmap, bounds):
    import matplotlib.colors as clr  # Importing the necessary library

    cmap = plt.get_cmap(cmap)  # Retrieving the colormap from matplotlib by name

    # Creating a list of colors in the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # Creating a new colormap with the same name, using the color list and the number of colors in the original colormap
    cmap = clr.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)

    # Creating a boundary norm for the colormap, which maps values to the nearest color in the colormap
    norm = clr.BoundaryNorm(bounds, cmap.N)

    # Setting the color to be used for values below the minimum value of the colormap
    cmap.set_under(color="white")

    # Setting the color to be used for values above the maximum value of the colormap
    cmap.set_over(color="white")

    # Returning the new colormap and boundary norm
    return cmap, norm


class EasyAnalyzer:
    def __init__(
        self, refmol="coordinates.pdb", reftic="tic.dat", refmodel="model.dat"
    ):
        self.refmol = Molecule(refmodel)
        self.reftic = TICApyemma(20).load(reftic)

        self.refmodel = Model(file=refmodel)
        self.mweights = computeWeights(refmodel)

    def plot_ref_tic(self):
        levels = np.arange(0, 7.6, 1.5)
        cmap = "viridis"
        dimx, dimy = 0, 1
        refstates = list(range(self.refmol.macronum))[::-1]

        plt.figure(figsize=[10, 10])
        plotContour(
            np.concatenate(self.refmol.data.dat),
            self.mweights,
            levels,
            dimx=dimx,
            dimy=dimy,
            cmap=cmap,
        )
        cbar = plt.colorbar()
        cbar.ax.tick_params()
        cbar.ax.get_yaxis().labelpad = 20
        cbar.ax.set_ylabel("kcal/mol", rotation=270)
        # plotstates(refmodel, states=refstates, dimx=dimx, dimy=dimy, cmap='tab20')
        # plt.legend(fontsize=10, bbox_to_anchor=(1.25, 1), loc='upper left')
        plt.xlabel(f"TICA dim {dimx}", size=16)
        plt.ylabel(f"TICA dim {dimy}", size=16)
        plt.title(
            "Reference MD simulations - {:.1f}µs".format(
                np.concatenate(self.refmodel.data.dat).shape[0] * 0.1
            )
        )
        plt.show()
