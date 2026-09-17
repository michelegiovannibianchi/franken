import dataclasses

import torch
import torch.nn as nn

from franken.config import LESConfig
import franken.data.base
from franken.data.base import Configuration
from franken.utils.derivatives import _forces_bwdad_helper, _forces_stress_bwdad_helper #function to derive energy w.r.t. to space or displacement to get force amd stress via autograd
from franken.utils.misc import sanitize_init_dict
from franken.les.ewald import Ewald


def initialize_les(les_config: LESConfig, feature_dim: int):# Converts the config into a dictionary and removes keys that LESHead does not need.
    les_params = sanitize_init_dict(LESHead, dataclasses.asdict(les_config))
    return LESHead(input_dim=feature_dim, **les_params)


class LESHead(nn.Module):
    """
    Predicts atomic charges from GNN embeddings and computes
    electrostatic energy with LES.

    Output:
        total_energy : (1,)
    """

    def __init__(
        self,
        input_dim: int,
        n_layers: int = 3,
        hidden_dim: tuple[int, ...] = (32, 16),
        dl: float = 2.0,
        sigma=1.0,
        les_output_scale: float = 0.1,  
        add_linear_nn: bool = True,
    ):
        super().__init__()
        self.les_output_scale = les_output_scale

        # Build the MLP
        if isinstance(hidden_dim, int):
            hidden_dim = (hidden_dim,)
        if len(hidden_dim) == 1: #last layer
            n_hidden = [hidden_dim[0]] * (n_layers - 1)
        elif len(hidden_dim) != n_layers - 1:
            raise ValueError(
                f"Expected {n_layers - 1} dimensions for the MLP. Found dimensions {hidden_dim}."
            )
        else:
            n_hidden = list(hidden_dim)
        n_neurons = [input_dim] + n_hidden + [1] #e.g fom mace-mp0 n_neurons = [128,32,16,1]
        layers = []
        for i in range(n_layers - 1):# here assemble NN
            layers.append(nn.Linear(n_neurons[i], n_neurons[i + 1])) # for i=0 create Linear(128 → 32) #for i=2 Linear(32 → 16)
            layers.append(nn.SiLU(inplace=True))# Add SILU as activation funcion.
        layers.append(nn.Linear(n_neurons[-2], n_neurons[-1]))# add Linear(16 =>1) Output is a scalar per atom
        self.outnet = nn.Sequential(*layers)
        self.linear_nn = None
        if add_linear_nn: # eventually add a linear output such that q = W*h +b + non-linear MLIP output
            self.linear_nn = nn.Linear(input_dim, 1)

        self.ewald = Ewald(
            dl=dl,
            sigma=sigma,
        )

        for module in self.modules(): #initialisation
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01) #weight
                if module.bias is not None: #bias of MLIP
                    nn.init.zeros_(module.bias)

    def predict(
        self,
        atom_features: torch.Tensor, #per-atom embeddings produced by the GNN.
        data: Configuration, #Contains the physical system information: atom positions,cell vectors,possibly species and other metadata
        displacement: torch.Tensor | None, #required only for stress calculation
        targets: list[str], #e.g. force stress
    ):
        compute_force = franken.data.base.FORCES_TARGET_KEY in targets
        compute_stress = franken.data.base.STRESS_TARGET_KEY in targets  #equivalent to compute_force = ("forces" in targets) True/False
        compute_LES_charges= franken.data.base.LES_CHARGES_TARGET_KEY in targets
        energies, les_charges = self(atom_features, data)
        
        if compute_stress:
            assert displacement is not None
            forces, stress = _forces_stress_bwdad_helper(energies, displacement, data)
            return {
                franken.data.base.FORCES_TARGET_KEY: forces.detach(),
                franken.data.base.STRESS_TARGET_KEY: stress.detach(),
                franken.data.base.ENERGY_TARGET_KEY: energies.detach(),
            }
        elif compute_force:
            forces, stress = _forces_bwdad_helper(energies, data)
            return {
                franken.data.base.FORCES_TARGET_KEY: forces.detach(),
                franken.data.base.ENERGY_TARGET_KEY: energies.detach(),
            }
        elif compute_LES_charges:
            forces, stress = _forces_bwdad_helper(energies, data)
            
            return {
                    franken.data.base.FORCES_TARGET_KEY: forces.detach(),
                    franken.data.base.ENERGY_TARGET_KEY: energies.detach(),
                    franken.data.base.LES_CHARGES_TARGET_KEY: les_charges.detach(),
            }
        else:
            return {
                franken.data.base.ENERGY_TARGET_KEY: energies.detach(),
            }

    def forward(
        self,
        atom_features: torch.Tensor,
        atom_pos: torch.Tensor,
        configuration: Configuration,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        atom_features : Tensor
            Tensor of atomic features obtained through a GNN. Should have shape
            `[n_atoms, hidden_dim]`.
        configuration : Configuration

        Returns
        -------
        energy : tensor shape (1,)
        charge : tensor shape (1,)
        """
        # predict atomwise contributions

        y = self.outnet(atom_features)
        if self.linear_nn is not None:
            y = y + self.linear_nn(atom_features)#Output is essential a linear contribution + a non-linear contribution
        y = y * self.les_output_scale # scale down charge to avoid large electrostatic energy, above all when parameter are randomly initialised
        cell = configuration.cell
        assert cell is not None
        if cell.dim() == 2:
            cell = cell.unsqueeze(0)
        E_lr, q_induced, u_induced = self.ewald(
            q=y,
            r=atom_pos,
            cell=cell,
            batch=configuration.batch_ids,
        )
        return E_lr, y
