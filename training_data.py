"""Training data preparations."""
import functools
import multiprocessing

import numpy as np
import pandas as pd
import pymeshlab as pml
import yaml


N_SAMPLING_POINTS = 8096

def read_obj(path) -> pml.MeshSet:
    ms = pml.MeshSet()
    ms.load_new_mesh(path)
    return ms


def read_feat(path) -> dict:
    with open(path, "r") as fi:
        feat = yaml.load(fi, yaml.CLoader)
    return feat


def sample_point_cloud(ms: pml.MeshSet) -> pd.DataFrame:
    ms.set_current_mesh(0)
    ms.generate_sampling_montecarlo(samplenum=N_SAMPLING_POINTS)

    pcloud = pd.DataFrame(
        ms.current_mesh().vertex_matrix(),
        columns=["x", "y", "z"]
    )
    return pcloud


def mark_edges_and_corners(mesh: pml.Mesh, feat: dict) -> pd.DataFrame:
    orig_points = mesh.vertex_matrix()

    curv_info = pd.DataFrame(feat["curves"])
    edge_point_idxs = curv_info.vert_indices.explode().astype(int)

    curv = (
        edge_point_idxs
            .rename("idx").to_frame()
            .pipe(_mark_corner)
            .pipe(_merge_coords, orig_points=orig_points)
    )

    return curv


def transfer_labels(curv: pd.DataFrame, pcloud: pd.DataFrame) -> pd.DataFrame:
    curv_ = curv.drop_duplicates(subset=["idx"])

    pcloud_df_idxs = curv_.apply(_transfer_gt_labels, pcloud=pcloud, axis=1)

    pcloud_ = (
        curv_
            .assign(pcloud_df_idx=pcloud_df_idxs)
            .merge(pcloud,
                   how="right",
                   left_on="pcloud_df_idx", right_index=True,
                   suffixes=("_orig", None))
            .drop(columns=["idx", "pcloud_df_idx"])
            .assign(is_edge=lambda df: df.is_corner.notna(),
                    is_corner=lambda df: df.is_corner == True)
    )
    return pcloud_


def _mark_corner(edge: pd.DataFrame):
    val_counts = edge.idx.value_counts()
    return edge.assign(is_corner=edge.idx.map(lambda i: val_counts[i] > 1))


def _merge_coords(edge, orig_points):
    return edge.assign(
        x=edge.idx.map(lambda i: orig_points[i][0]),
        y=edge.idx.map(lambda i: orig_points[i][1]),
        z=edge.idx.map(lambda i: orig_points[i][2]),
    )


def _transfer_gt_labels(row: pd.Series, pcloud: pd.DataFrame):
    dist_vects = pcloud[["x", "y", "z"]].values - row[["x", "y", "z"]].values
    dist = np.square(dist_vects).sum(axis=1)
    return dist.argmin()
