# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
def optimize_cell_width():
    from IPython.display import display, HTML
    display(HTML("<style>.container { width:100% !important; }</style>"))

if __name__ == '__main__':
    optimize_cell_width()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
# # %matplotlib notebook

# %%
# !python -V

# %% [markdown]
# ## Download Data
# #### `.obj` files

# %%
# # !mkdir -p data/obj
# # only the first chunk
# # !head -n1 data/obj_v00.txt | xargs -n 2 -P 8 sh -c 'wget --no-check-certificate $0 -O data/obj/$1'

# %% [markdown]
# #### Corresponding feat files

# %%
# # !mkdir -p data/feat
# # only the first chunk
# # !head -n1 data/feat_v00.txt | xargs -n 2 -P 8 sh -c 'wget --no-check-certificate $0 -O data/feat/$1'

# %% [markdown]
# ## Some example shapes in ABC dataset

# %%
# !ls 

# %%
# !pip install pymeshlab
# # !pip install polyscope  # optional to render the MeshSet state

# %%
def describe_mesh(mesh):
    print("# faces", mesh.face_number())
    print("# edges", mesh.edge_number())
    print("# vertices", mesh.vertex_number())


# %%
EX_STEP_PATH = "data/00000050_80d90bfdd2e74e709956122a_step_000.step"
EX_OBJ_PATH = "data/00000050_80d90bfdd2e74e709956122a_trimesh_000.obj"
# EX_OBJ_PATH = "data/obj/00008338/00008338_75b44178dbe14c99b75a0738_trimesh_008.obj"
EX_FEAT_PATH = "data/00000050_80d90bfdd2e74e709956122a_features_000.yml"
# EX_FEAT_PATH = "data/feat/00008338/00008338_75b44178dbe14c99b75a0738_features_008.yml"

# %% [markdown]
# TODO
# - [ ] *decide which to use to load .step file, blender or gmsh
#   - blender does not support .step file by default
#   - [ ] *take a look at gmsh
#   - [x] **but maybe using .obj + .feat is sufficient?**
#     - looks feasible according to ABC paper
# - [x] figure out how to iter through surfaces in a CAD model (loaded from a .obj file)
#   - `pymeshlab` is promising
#     - montecarlo (poisson disk bit slow although looked better)
# - [x] figure out how to define ground truth from obj and/or feat.yaml file
#   - YL: 'Nearest neighbour' in PIE paper probably meant using the vertices defined in feature file to find the 1-NN in sampled point cloud

# %%
import pymeshlab as pml

ms = pml.MeshSet()

ms.load_new_mesh(EX_OBJ_PATH)
ms.number_meshes()

# %%
mesh = ms.current_mesh()
describe_mesh(mesh)

# %%
orig_points = mesh.vertex_matrix()
type(orig_points), len(orig_points)

# %%
import random

SAMPLE_K = 8096

sampled_orig_points = np.array(random.sample(list(orig_points), SAMPLE_K))
sampled_orig_points[:3]

# %%
MPL_FIG_SIZE = (12, 12)

def plot_point_cloud(matrix, ax=None, **kwargs):
    xs = matrix[:, 0]
    ys = matrix[:, 1]
    zs = matrix[:, 2]
    
    if ax is None:
        __, ax = plt.subplots(
            figsize=MPL_FIG_SIZE,
            subplot_kw=dict(projection="3d"),
        )
        ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))

    kwargs.setdefault("s", 1)
    ax.scatter(xs, ys, zs, **kwargs)
    return ax


# %%
plot_point_cloud(sampled_orig_points)

# %% [markdown]
# ### Sampling

# %%
ms.set_current_mesh(0)
ms.generate_sampling_poisson_disk(
    samplenum=SAMPLE_K,
    exactnumflag=True, # within 0.5% of samplenum, slower
)
ms.number_meshes()

# %%
mesh_pc_pd = ms.current_mesh()
describe_mesh(mesh_pc_pd)

# %%
plot_point_cloud(mesh_pc_pd.vertex_matrix())

# %%
ms.set_current_mesh(0)
ms.generate_sampling_montecarlo(
    samplenum=SAMPLE_K,
)
ms.number_meshes()

# %%
mesh_pc_mc = ms.current_mesh()
describe_mesh(mesh_pc_mc)

# %%
plot_point_cloud(mesh_pc_mc.vertex_matrix())

# %% [markdown]
# #### Are MC sampled points different from orig vertices?
# YES for all

# %%
_orig_verts = pd.DataFrame(
    ms.mesh(0).vertex_matrix(),
    columns=["x", "y", "z"])
print(_orig_verts.shape)

_sampled_pts = pd.DataFrame(
    mesh_pc_mc.vertex_matrix(),
    columns=["x", "y", "z"])
print(_sampled_pts.shape)

_orig_verts.drop_duplicates().shape[0] + _sampled_pts.drop_duplicates().shape[0]

# %%
pd.concat([_orig_verts, _sampled_pts]).drop_duplicates().shape

# %% [markdown]
# ## Check `feat.yaml`

# %%
import yaml

def read_yaml(path):
    with open(path, "r") as fi:
        content = yaml.safe_load(fi)
    return content


# %%
# %%time
feat = read_yaml(EX_FEAT_PATH)
type(feat), len(feat)

# %%
curves = feat['curves']
type(curves), len(curves)

# %%
curv = pd.DataFrame(curves)
curv.shape

# %% [markdown]
# ### Edge points

# %%
curve_point_idxs = curv.vert_indices.explode().astype(int)
curve_point_idxs.shape

# %%
print("Orig # points:", len(orig_points))
curve_point_idxs.describe()

# %%
curve_point_idxs.sort_values().diff().describe()
# idx starts from 0 and increments by 1 at most, i.e. continous

# %% [markdown]
# #### Max curve point index << # orig points; what gives?
# See PC plot below
# - curve points had the smallest indices
# - correctness of indices is also verified!

# %%
curve_points = orig_points[curve_point_idxs.drop_duplicates()]
curve_points.shape

# %%
plot_point_cloud(curve_points, c="g")

# %%
print(curv.loc[0, "type"])

one_curve_points = orig_points[pd.Series(curv.loc[0, "vert_indices"])]
plot_point_cloud(one_curve_points)

# one_curve_points_adjusted = orig_points[pd.Series(curv.loc[0, "vert_indices"])-1]
# plot_point_cloud(one_curve_points_adjusted)

# %% [markdown]
# ### Corner points

# %%
curve_point_idxs.value_counts().value_counts().sort_index()
# having > 1 indicates corner points
# - correctness verified; see enhanced PC plot below

# %%
def mark_corner(df: pd.DataFrame):
    idx_v_cnt = df.idx.value_counts()
    return df.assign(is_corner=df.idx.map(lambda i: idx_v_cnt.loc[i] > 1))


# %%
def merge_coords(df, orig_points):
    return df.assign(
        x=df.idx.map(lambda i: orig_points[i][0]),
        y=df.idx.map(lambda i: orig_points[i][1]),
        z=df.idx.map(lambda i: orig_points[i][2]),
    )


# %%
curve_pts = (
    curve_point_idxs
        .rename("idx").to_frame()
        .pipe(mark_corner)
        .pipe(merge_coords, orig_points=orig_points)
)
curve_pts.shape

# %%
curve_pts

# %%
corner_point_idxs = curve_pts.loc[curve_pts.is_corner ,"idx"]
corner_points = orig_points[corner_point_idxs.drop_duplicates()]
corner_points.shape

# %%
ax = plot_point_cloud(curve_points, c="g")
ax = plot_point_cloud(corner_points, ax=ax, s=3, c="r")

# %% [markdown]
# ## Nearest Neighbour assignment/transfer for GT labels
# ### Prep sampled data

# %%
sampled_pts = (
    pd.DataFrame(mesh_pc_mc.vertex_matrix(), columns=["x", "y", "z"])
)
sampled_pts.shape

# %%
sampled_pts

# %%
curve_pts


# %%
# # !pip install line_profiler

# %%
# %load_ext line_profiler

# %%
def transfer_gt_labels(sampled_pts, row: pd.Series):
    dist_vects = sampled_pts[["x", "y", "z"]].values - row[["x", "y", "z"]].values
    dist = np.square(dist_vects).sum(axis=1)
    return dist.argmin()


# %%
# %lprun -f transfer_gt_labels transfer_gt_labels(curve_pts.iloc[0], sampled_pts)

# %%
curve_pts = curve_pts.reset_index(drop=True)
curve_pts

# %%
# single process took:
"""
CPU times: user 7min 13s, sys: 253 ms, total: 7min 14s
Wall time: 7min 14s
"""

# %%
# %%time
# default chunksize: 3.5min

import multiprocessing
import functools

find_nearest = functools.partial(transfer_gt_labels, sampled_pts)

curve_pts_ = curve_pts.drop_duplicates(subset=["idx"])
curve_df_rows = (row for __, row in curve_pts_.iterrows())

with multiprocessing.Pool() as pool:
    sampled_df_idxs = pool.map(find_nearest, curve_df_rows, chunksize=100)
type(sampled_df_idxs), len(sampled_df_idxs)

# %%
sampled_pts_ = (
    curve_pts_
        .assign(sampled_df_idx=sampled_df_idxs)
        .merge(sampled_pts,
               how="right",
               left_on="sampled_df_idx", right_index=True,
               suffixes=("_orig", None))
        .drop(columns=["idx", "sampled_df_idx"])
        .assign(is_edge=lambda df: df.is_corner.notna(),
                is_corner=lambda df: df.is_corner == True)
)
sampled_pts_.shape

# %%
_edge_points = sampled_pts_.query("is_edge == True")[["x", "y", "z"]].values
ax = plot_point_cloud(_edge_points, c="g")

_corner_ponits = sampled_pts_.query("is_corner == True")[["x", "y", "z"]].values
ax = plot_point_cloud(_corner_ponits, ax=ax, s=5, c="r")

# %% [markdown]
# ## Test training_data.py

# %%
import training_data

# %%
# %%time
cad_model = training_data.read_obj(EX_OBJ_PATH)
feat = training_data.read_feat(EX_FEAT_PATH)

pcloud = training_data.sample_point_cloud(cad_model)

curv = training_data.mark_edges_and_corners(cad_model.mesh(0), feat)

pcloud_ = training_data.transfer_labels(curv, pcloud)

# %%
pcloud_.sample(10)

# %%
_edge_points = pcloud_.query("is_edge == True")[["x", "y", "z"]].values
ax = plot_point_cloud(_edge_points, c="g")

_corner_ponits = pcloud_.query("is_corner == True")[["x", "y", "z"]].values
ax = plot_point_cloud(_corner_ponits, ax=ax, s=5, c="r")

# %% [markdown]
# ## MISC

# %%
