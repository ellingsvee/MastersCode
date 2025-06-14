from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import rpy2.robjects as ro
import scipy.sparse as sparse
import torch
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from scipy.spatial import Delaunay
from shapely.geometry import Polygon
from shapely.vectorized import contains

from utils.scipy_torch_conversion import scipy_csc_to_torch_coo


class Mesh:
    def __init__(self):
        self.tri: Delaunay = None
        self.mask: np.ndarray = None
        numpy2ri.activate()
        fmesher = importr("fmesher")

    def create_mesh(
        self,
        max_edge: np.ndarray,
        loc_domain: np.ndarray = None,
        loc: np.ndarray = None,
        cutoff: np.ndarray = None,
        offset: np.ndarray = None,
    ):
        # Using the fmesher package to create a mesh
        if np.where(max_edge == 0)[0].size > 0:
            raise ValueError("max_edge cannot be zero.")

        # If locations are provided, use them to create the mesh
        if loc is None:
            ro.r("loc <- NULL")
        else:
            loc = np.array(loc)
            loc_domain = None
            ro.r.assign("loc", loc)

        # If loc_domain is provided, use it to create the mesh
        if loc_domain is not None:
            loc_domain = np.array(loc_domain)
            self.A = np.max(loc_domain[:, 0])
            self.B = np.max(loc_domain[:, 1])
            ro.r.assign("loc_domain", loc_domain)
        else:
            ro.r("loc_domain <- NULL")

        max_edge = np.array(max_edge)
        ro.r.assign("max_edge", max_edge)

        if offset is None:
            ro.r("offset <- NULL")
        else:
            offset = np.array(offset)
            offset[np.where(offset == 0)] = 1e-10
            ro.r.assign("offset", offset)
            ro.r("offset <- as.numeric(offset)")

        if cutoff is None:
            ro.r("cutoff <- NULL")
        else:
            ro.r.assign("cutoff", np.array(cutoff))
            ro.r("cutoff <- as.numeric(cutoff)")

        # Create the mesh using the fmesher package
        if loc is None:
            ro.r(
                """
                max_edge <- as.numeric(max_edge)
                m <- fm_mesh_2d(loc.domain = loc_domain, max.edge=max_edge, cutoff = cutoff, offset=offset)
                """
            )
        else:
            ro.r(
                """
                max_edge <- as.numeric(max_edge)
                m <- fm_mesh_2d(loc=loc, max.edge=max_edge, cutoff = cutoff, offset=offset)
                """
            )

        # Saving the parameters
        self.loc = loc
        self.loc_domain = loc_domain
        self.max_edge = max_edge
        self.cutoff = cutoff
        self.offset = offset

        # Retrieving the points from the fmesher object
        m = ro.r("m")
        self.fm_vertices = np.array(m.rx2("loc"))
        self.fm_simplices = np.array(ro.r("m$graph$tv - 1L"))

        self.fm_int_bndry = np.array(ro.r("m$segm$int$idx - 1L"))
        self.fm_ext_bndry = np.array(ro.r("m$segm$bnd$idx - 1L"))

        self.int_bndry_pts = self.fm_vertices[:, :2][self.fm_int_bndry].reshape(-1, 2)
        self.ext_bndry_pts = self.fm_vertices[:, :2][self.fm_ext_bndry].reshape(-1, 2)

        # Create the boundary polygons. Unsure where these are used
        self.int_boundary_polygon = Polygon(self.int_bndry_pts)
        self.loc_domain_polygon = Polygon(self.loc_domain)
        self.true_loc_domain_polygon = Polygon(self.loc_domain)

        # Calculating the Delaunay triangulation
        self.tri = Delaunay(self.fm_vertices[:, :2], qhull_options="Qx")
        self.vertices = self.tri.points
        self.n = self.vertices.shape[0]
        self.simplices = self.tri.simplices
        self.n_simplices = self.simplices.shape[0]
        self.centroids = np.mean(self.vertices[self.simplices], axis=1)

        # Calculate the bounding box
        self.A1 = np.min(self.vertices[:, 0])
        self.A2 = np.max(self.vertices[:, 0])
        self.B1 = np.min(self.vertices[:, 1])
        self.B2 = np.max(self.vertices[:, 1])
        self.A = self.A2 - self.A1
        self.B = self.B2 - self.B1

        # Create a mask to identify points inside the domain
        self.mask = self.get_inside_mask(self.vertices)
        self.simplices_mask = self.get_simplices_mask()

        # Raise an error if the mesh is degenerate
        self.check_non_degenerate()

    def create_mesh_from_fmesher_mesh(
        self,
        fm_vertices: np.ndarray,
        fm_int_bndry: np.ndarray,
        fm_ext_bndry: np.ndarray,
    ):
        self.fm_vertices = fm_vertices
        self.fm_int_bndry = fm_int_bndry
        self.fm_ext_bndry = fm_ext_bndry

        self.int_bndry_pts = self.fm_vertices[:, :2][self.fm_int_bndry].reshape(-1, 2)
        self.ext_bndry_pts = self.fm_vertices[:, :2][self.fm_ext_bndry].reshape(-1, 2)

        # Create the boundary polygons. Unsure where these are used
        self.int_boundary_polygon = Polygon(self.int_bndry_pts)
        self.ext_boundary_polygon = Polygon(self.ext_bndry_pts)

        self.loc_domain = self.int_bndry_pts
        self.loc_domain_polygon = self.int_boundary_polygon
        # self.loc_domain_polygon = self.int_boundary_polygon
        # self.loc_domain_polygon = Polygon(self.loc_domain)

        self.tri = Delaunay(self.fm_vertices[:, :2], qhull_options="Qx")

        self.vertices = self.tri.points
        self.n = self.vertices.shape[0]
        self.simplices = self.tri.simplices
        self.n_simplices = self.simplices.shape[0]
        self.centroids = np.mean(self.vertices[self.simplices], axis=1)

        # Calculate the bounding box
        self.A1 = np.min(self.vertices[:, 0])
        self.A2 = np.max(self.vertices[:, 0])
        self.B1 = np.min(self.vertices[:, 1])
        self.B2 = np.max(self.vertices[:, 1])
        self.A = self.A2 - self.A1
        self.B = self.B2 - self.B1

        # Create a mask to identify points inside the domain
        self.mask = self.get_inside_mask(self.vertices)
        self.simplices_mask = self.get_simplices_mask()

        # Raise an error if the mesh is degenerate
        self.check_non_degenerate()

    def check_non_degenerate(self):
        v0 = self.vertices[self.simplices[:, 0]]
        v1 = self.vertices[self.simplices[:, 1]]
        v2 = self.vertices[self.simplices[:, 2]]
        e0 = v2 - v1
        e1 = v0 - v2
        cross_products = np.cross(e0, e1)
        areas = 0.5 * np.abs(cross_products)
        if len(np.where(areas == 0)[0]) > 0:
            raise ValueError("Degenerate mesh generated.")

    def get_inside_mask(self, points: np.ndarray) -> np.ndarray:
        x, y = points[:, 0], points[:, 1]  # Extract x and y coordinates of points
        try:
            mask = contains(self.true_loc_domain_polygon, x, y)
        except:
            mask = contains(self.true_loc_domain_polygon, x, y)
            # mask = contains(self.int_boundary_polygon, x, y)

        return mask

    def get_simplices_mask(self) -> np.ndarray:
        v0 = self.vertices[self.simplices[:, 0]]
        v1 = self.vertices[self.simplices[:, 1]]
        v2 = self.vertices[self.simplices[:, 2]]
        v_avg = (v0 + v1 + v2) / 3
        x, y = v_avg[:, 0], v_avg[:, 1]  # Extract x and y coordinates of points
        try:
            mask = contains(self.loc_domain_polygon, x, y)
        except:
            mask = contains(self.int_boundary_polygon, x, y)

        return mask

    def create_observation_matrix(
        self,
        locs: np.ndarray,
    ) -> torch.Tensor:
        # Find the simplex (triangle) each point is in
        simplices = self.tri.find_simplex(locs)

        # Barycentric coordinates of each point with respect to its simplex
        transform = self.tri.transform[simplices]
        delta = locs - transform[:, 2]
        bary_coords = np.einsum("ijk,ik->ij", transform[:, :2, :], delta)
        bary_coords = np.c_[bary_coords, 1 - bary_coords.sum(axis=1)]

        # Create the sparse matrix
        A = sparse.csc_matrix(
            (
                bary_coords.ravel(),
                (
                    np.repeat(np.arange(len(locs)), 3),
                    self.tri.simplices[simplices].ravel(),
                ),
            ),
            shape=(len(locs), self.tri.npoints),
        )
        return scipy_csc_to_torch_coo(A)

    def interpolate_points(self, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        # Create projector matrix
        A = self.create_observation_matrix(points, return_torch=False)
        # Interpolate values using the projector matrix
        interpolated_values = A @ weights.T
        return interpolated_values

    def plot(
        self,
        weights: Union[torch.tensor, np.ndarray] = None,
        cmap: str = "viridis",
        vmin: float = None,
        vmax: float = None,
        xlim: float = None,
        ylim: float = None,
        symmetric_cbar: bool = True,
        colorbar: bool = True,
        plot_loc_domain: bool = False,
        only_inside_loc_domain: bool = False,
        save_fig_name: Optional[str] = None,
        close_after_save: bool = True,
        title: Optional[str] = None,
        n_ticks=5,
        linewidth: float = 2,
    ):
        mask = np.ones(self.vertices.shape[0], dtype=bool)

        if only_inside_loc_domain:
            if xlim is None and ylim is None:
                xmin = self.loc_domain[:, 0].min()
                xmax = self.loc_domain[:, 0].max()
                ymin = self.loc_domain[:, 1].min()
                ymax = self.loc_domain[:, 1].max()
                xlim = (xmin, xmax)
                ylim = (ymin, ymax)

        if weights is not None:
            if only_inside_loc_domain:
                mask = self.mask
                weights[~mask] = np.nan

            if symmetric_cbar:
                # max_abs_value = np.max(np.abs(weights[mask]))
                max_abs_value = np.max(np.abs(weights[mask]))
                vmin = -max_abs_value
                vmax = max_abs_value

            else:
                min_value = np.min(weights[mask])
                max_value = np.max(weights[mask])
                if vmax is None:
                    vmax = max_value
                if vmin is None:
                    vmin = min_value

            plt.tripcolor(
                self.vertices[:, 0],
                self.vertices[:, 1],
                self.simplices,
                weights,
                shading="gouraud",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
        else:
            plt.triplot(
                self.vertices[:, 0],
                self.vertices[:, 1],
                self.simplices,
                color="black",
                linewidth=0.5,
            )

        if plot_loc_domain:
            plt.plot(
                self.loc_domain[:, 0],
                self.loc_domain[:, 1],
                color="black",
                linewidth=linewidth,
            )

        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlim(xlim)
        plt.ylim(ylim)

        if title is not None:
            plt.title(title)

        if weights is not None and colorbar:
            plt.colorbar()

        if save_fig_name is not None:
            fig = plt.gcf()
            fig.patch.set_facecolor("white")  # Figure background
            plt.gca().set_facecolor("white")  # Axes background

            plt.savefig(
                save_fig_name,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="white",  # This is often missing but important for PDFs
                pad_inches=0.01,
            )

            if close_after_save:
                plt.close()
            else:
                plt.show()
        else:
            plt.show()
