import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import copy
from typing import Optional

import torch

from spde.inference import Inference
from spde.matrices import Matrices
from spde.mesh import Mesh
from spde.parameters import Parameters
from spde.utils import Utils


class SPDEModel:
    def __init__(self, mesh: Optional[Mesh] = None):
        if mesh is not None:
            self.set_mesh(mesh)

    def set_mesh(self, mesh: Mesh) -> None:
        self.mesh = mesh
        self.params = Parameters(mesh=mesh)
        self.matrices = Matrices(mesh=mesh, params=self.params)
        self.inference = Inference(
            mesh=mesh, params=self.params, matrices=self.matrices
        )
        self.utils = Utils(
            mesh=mesh,
            params=self.params,
            matrices=self.matrices,
            inference=self.inference,
        )

    def refresh(self) -> None:
        if self.params.anisotropy:
            self.matrices.G = self.matrices.assemble_G(
                vx_vals=self.params.vx,
                vy_vals=self.params.vy,
                anisotropy=self.params.anisotropy,
            )
        self.matrices.compute_rational_approx_matrices()

    def gen_and_plot_realization(
        self,
        plot_loc_domain: bool = True,
        only_inside_loc_domain: bool = True,
        save_fig_name: Optional[str] = None,
    ) -> None:
        realization = self.inference.simulate_field()
        self.mesh.plot(
            weights=realization[:, 0],
            plot_loc_domain=plot_loc_domain,
            only_inside_loc_domain=only_inside_loc_domain,
            save_fig_name=save_fig_name,
        )
