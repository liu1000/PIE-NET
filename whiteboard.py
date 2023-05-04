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
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
# # %matplotlib notebook

# %%
# !python -V

# %% [markdown]
# # Training Data
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
# !ls data

# %%
DATA_DIR = pathlib.Path("./data/")

EX_STEP_PATH = "data/00000050_80d90bfdd2e74e709956122a_step_000.step"
EX_OBJ_PATH = "data/00000050_80d90bfdd2e74e709956122a_trimesh_000.obj"
# EX_OBJ_PATH = "data/obj/00008338/00008338_75b44178dbe14c99b75a0738_trimesh_008.obj"
EX_FEAT_PATH = "data/00000050_80d90bfdd2e74e709956122a_features_000.yml"
# EX_FEAT_PATH = "data/feat/00008338/00008338_75b44178dbe14c99b75a0738_features_008.yml"

# %%
def describe_mesh(mesh):
    print("# faces", mesh.face_number())
    print("# edges", mesh.edge_number())
    print("# vertices", mesh.vertex_number())


# %%
MPL_FIG_SIZE = (10, 10)

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


# %% [markdown]
# TODO
# - [x] *decide which to use to load .step file, blender or gmsh
#   - blender does not support .step file by default
#   - [x] *take a look at gmsh
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
plot_point_cloud(sampled_orig_points)

# %% [markdown]
# ### Sampling

# %%
# %%time
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
# Q: Are MC sampled points different from orig vertices? YES for all

# %%
_orig_verts = pd.DataFrame(
    ms.mesh(0).vertex_matrix(),
    columns=["x", "y", "z"])
print(_orig_verts.shape)

_sampled_pts = pd.DataFrame(
    mesh_pc_mc.vertex_matrix(),
    columns=["x", "y", "z"])
print(_sampled_pts.shape)

_orig_verts.merge(_sampled_pts).shape

# %% [markdown]
# ### Sampling quick study (post-hoc)
# TODO
# - [x] :sample 1000 ABC models
# - [x] run poissong disk
# - [x] run MC
# - [x] compare speed
# - [x] compare some summary stats
#   - [x] number of sampled points
#   - [x] 1-nearest neighbor distance, mean/std (need to normalize metrics to compare cross different models)
#     - normalized by range
# - [x] compare some qualititive plots

# %%
import itertools
import random
import time

import training_data

random.seed(2023)


def sample_some_abc_obj_paths(n=1000):
    obj_dir_paths = [x for x in pathlib.Path('data/obj').iterdir() if x.is_dir()]
    obj_file_paths = itertools.chain.from_iterable(p.glob("*.obj") for p in obj_dir_paths)
    return random.sample(list(obj_file_paths), n)

sample_some_abc_obj_paths(3)


# %%
def read_run_sampling(path, sampling="mc"):
    ms = training_data.read_obj(path)
    ms.set_current_mesh(0)
    
    t0 = time.time()

    if sampling == "mc":
        ms.generate_sampling_montecarlo(samplenum=training_data.N_SAMPLING_POINTS)
    elif sampling == "poisson-disk":
        ms.generate_sampling_poisson_disk(samplenum=training_data.N_SAMPLING_POINTS)
    elif sampling == "poisson-disk-strict":
        ms.generate_sampling_poisson_disk(samplenum=training_data.N_SAMPLING_POINTS, exactnumflag=True)
    else:
        raise ValueError(f"Unsupported sampling: {sampling}")

    t1 = time.time()
    seconds = t1 - t0

    pcloud = pd.DataFrame(
        ms.current_mesh().vertex_matrix(),
        columns=["x", "y", "z"]
    )
    return pcloud, seconds


# %%
sampled_abc_obj_paths = sample_some_abc_obj_paths(1000)

# %% [markdown]
# #### speed of sampling

# %%
mc_pclouds, mc_times = [], []
for path in sampled_abc_obj_paths:
    pcloud, seconds = read_run_sampling(path, "mc")
    mc_pclouds.append(pcloud)
    mc_times.append(seconds)

sum(mc_times)

# %%
pdisk_pclouds, pdisk_times = [], []
for path in sampled_abc_obj_paths:
    pcloud, seconds = read_run_sampling(path, "poisson-disk")
    pdisk_pclouds.append(pcloud)
    pdisk_times.append(seconds)

sum(pdisk_times)

# %%
strict_pdisk_pclouds, strict_pdisk_times = [], []
for path in sampled_abc_obj_paths:
    pcloud, seconds = read_run_sampling(path, "poisson-disk-strict")
    strict_pdisk_pclouds.append(pcloud)
    strict_pdisk_times.append(seconds)

sum(strict_pdisk_times)


# %%
def compare_plots(mc_pcloud, pdisk_plcoud):
    __, axs = plt.subplots(1, 2, figsize=(20, 10), subplot_kw=dict(projection="3d"))

    plot_point_cloud(mc_pcloud.values, ax=axs[0])
    plot_point_cloud(pdisk_plcoud.values, ax=axs[1])

i = 0
compare_plots(mc_pclouds[i], pdisk_pclouds[i])

# %%
compare_plots(mc_pclouds[0], strict_pdisk_pclouds[0])

# %%
compare_plots(mc_pclouds[871], strict_pdisk_pclouds[871])


# %% [markdown]
# *Q*: Why do Poisson-Disk based point clouds show structural details better (than the MC based)?
#
# *Discussion*: Point clouds sampled by the Poisson-Disk algorithm were less noisy, as shown by lower dispersions in the 1-NN summary stats below.
# **In other words, the space distribution of points were more regular. This made [edges and corners, the irregulars,](https://cecas.clemson.edu/~stb/ece847/internal/classic_vision_papers/attneave_1954.pdf) stand out more.**
# I would guess that PIE-NET perform better for point clouds sampled with Poisson-Disk.

# %% [markdown]
# #### Summary stats
# ##### # points

# %%
def number_of_points(pclouds):
    return pd.Series([len(df) for df in pclouds])

number_of_points(mc_pclouds).plot(kind="hist", bins=20)

# %%
number_of_points(pdisk_pclouds).plot(kind="hist", bins=20)

# %%
number_of_points(strict_pdisk_pclouds).plot(kind="hist", bins=20)

# %% [markdown]
# ##### 1nn distance

# %%
import multiprocessing


def _one_nn(row, orig_df):
    loo_df = orig_df[orig_df.x.ne(row.x) | orig_df.y.ne(row.y) | orig_df.z.ne(row.z)]
    assert len(loo_df) + 1 == len(orig_df), f"{len(loo_df)}, {len(orig_df)}"

    dist_vects = loo_df[["x", "y", "z"]].values - row[["x", "y", "z"]].values
    dist = np.square(dist_vects).sum(axis=1)
    return min(dist)


def add_1nn_dist(pcloud):
    pcloud["onn_dist"] = pcloud.apply(_one_nn, orig_df=pcloud, axis=1)
    return pcloud

# add_1nn_dist(mc_pclouds[1])
# mc_pclouds[1]


# %%
# %%time
# pool = multiprocessing.Pool()

# mc_pclouds_ = pool.map(add_1nn_dist, mc_pclouds)

# pool.close()
"""
CPU times: user 2.03 s, sys: 865 ms, total: 2.89 s
Wall time: 11min 15s
"""

# %%
# for i, pcloud in enumerate(mc_pclouds_):
#     pcloud.to_parquet(f"data/sampling_study_mc_pclouds/pc_{i}.parq")

pd.read_parquet("data/sampling_study_mc_pclouds/pc_879.parq")

# %% tags=[]
# pool = multiprocessing.Pool()

# strict_pdisk_pclouds_ = pool.map(add_1nn_dist, strict_pdisk_pclouds)

# pool.close()

# %%
# for i, pcloud in enumerate(strict_pdisk_pclouds_):
#     pcloud.to_parquet(f"data/sampling_study_strict_pdisk_pclouds/pc_{i}.parq")

# pd.read_parquet("data/sampling_study_strict_pdisk_pclouds/pc_879.parq")

# %%
####### Distribution of 1NN distance for one model
PCLOUD_IND = 871

mc_pclouds_[PCLOUD_IND].onn_dist.plot(kind="hist", bins=50, xlim=(0, 2.5), ylim=(0, 5000))

# %%
mc_pclouds_[PCLOUD_IND].onn_dist.std()

# %%
strict_pdisk_pclouds_[PCLOUD_IND].onn_dist.plot(kind="hist", bins=15, xlim=(0.0, 2.5), ylim=(0, 5000))

# %%
strict_pdisk_pclouds_[PCLOUD_IND].onn_dist.std()

# %%
###### Distribution of (normalized) dispersions of 1NN distances for 1000 models
mc_stds, spd_stds = [], []

for i in range(len(mc_pclouds_)):
    onn_dist_range = mc_pclouds_[i].onn_dist.max() - mc_pclouds_[i].onn_dist.min()
    # to normalize STDs of different models 

    mc_norm_std = mc_pclouds_[i].onn_dist.std() / onn_dist_range
    mc_stds.append(mc_norm_std)

    spd_norm_std = strict_pdisk_pclouds_[i].onn_dist.std() / onn_dist_range
    spd_stds.append(spd_norm_std)

len(mc_stds), len(spd_stds)

# %%
(
    pd.DataFrame({"MC 1NN std": mc_stds, "Poisson-Disk (strict) std": spd_stds})
        .melt(var_name="metric")
        .pipe((sns.displot, "data"),
              x="value", hue="metric", row="metric", aspect=2/1, height=3)
)

# %%

# %%

# %%

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

# %%
curv

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
# Q: Max curve point index << # orig points; what gives? See PC plot below
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
def plot_edges_and_corners(pcloud_, ax=None):
    _edge_points = pcloud_.query("is_edge == True")[["x", "y", "z"]].values
    ax = plot_point_cloud(_edge_points, ax=ax, c="g")

    _corner_ponits = pcloud_.query("is_corner == True")[["x", "y", "z"]].values
    ax = plot_point_cloud(_corner_ponits, ax=ax, s=5, c="r")
    return ax


# %%
plot_edges_and_corners(sampled_pts_)

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
pcloud_[pcloud_.curv_id.isin(pcloud_.curv_id.drop_duplicates().sample())]

# %%
plot_edges_and_corners(pcloud_)

# %%
pcloud_

# %% [markdown]
# ### Save pcloud

# %%
print(training_data._format_pcloud_filename(EX_FEAT_PATH))


# %%
# training_data.write_pcloud(pcloud_, EX_FEAT_PATH)

# %% [markdown]
# ## EDA for feat files
#
# Q: proportion of files with uncommon types of curves

# %%
def read_curve_type_stats(path) -> pd.Series:
    feat = training_data.read_feat(path)
    feat_info = pd.DataFrame(feat["curves"])

    return feat_info.type.value_counts()


# %%
# %%time
read_curve_type_stats(EX_FEAT_PATH)

# %%
FEAT_DIR = DATA_DIR / "feat"

feat_paths = sorted([path for path in FEAT_DIR.glob("**/*features*.yml")])
len(feat_paths)

# %% [markdown]
# ---
# digress: matching obj files all exist?
# YES

# %%
import re

def get_corresponding_obj_path(feat_path):
    path_id = feat_path.parent.name
    obj_paths = list((DATA_DIR / "obj" / path_id).glob("*.obj"))
    assert len(obj_paths) == 1, f"not 1-to-1 mapping for {path_id}"
    return obj_paths[0]

for feat_path in feat_paths:
    get_corresponding_obj_path(feat_path)

# %% [markdown]
# ---
# Info: **~30 min** to go over the entire feature yml files

# %%
# # %%time
# with multiprocessing.Pool() as pool:
#     curve_type_stats = pool.map(read_curve_type_stats, feat_paths)
# len(curve_type_stats)

# %%
curtype = (
    pd.DataFrame(curve_type_stats)
        .reset_index(drop=True)
)
curtype.shape

# %%
curtype.notna().sum()

# %%
common_curtype = (
    curtype[curtype[["Ellipse", "Other"]].isna().all(axis=1)]
)
common_curtype.shape

# %% [markdown]
# ## Generate point clouds

# %%
training_data.generate_one_pcloud(feat_paths[2])

# %%
loaded_pcloud = pd.read_parquet("data/pcloud/00000004_pcloud_points.parq")
loaded_pcloud.shape

# %%
__, axs = plt.subplots(
    1, 2,
    figsize=(20, 10),
    subplot_kw=dict(projection="3d"))

plot_point_cloud(loaded_pcloud[["x", "y", "z"]].values, ax=axs[0])

plot_edges_and_corners(loaded_pcloud, ax=axs[1])

# %% tags=[]
import tqdm

feat_paths_to_process = tqdm.tqdm(feat_paths)

with multiprocessing.Pool() as pool:
    pool.map(training_data.generate_one_pcloud, feat_paths_to_process, chunksize=10)

# 1.5 hrs

# %%

# %% [markdown]
# # Point classification (+ offset regression)
# TODO
# - [ ] :try https://github.com/dgriffiths3/pointnet2-tensorflow2
#   - [ ] if no good, *try https://github.com/charlesq34/pointnet2/pull/154
# - [ ] Look at TF records API
# - [ ] :check how to point cls with PointNet++
#   - loss?
# - [ ] :check how to corrd reg with PointNet++
#   - loss?
#     - maybe skip coord reg for now if adding too much complexity to the custom loss
#   
# CUDA TODO (ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/contents.html#)
# - [x] might need to upgrade nvidia driver to 525.60.13; UPGRADED

# %%
import pathlib

import tensorflow as tf

# %%
SCANNET_TRAIN_PATH = "pointnet2-tensorflow2/data/scannet_train.tfrecord"

# %%
raw_dataset = tf.data.TFRecordDataset(SCANNET_TRAIN_PATH)
raw_dataset

# %%
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

# %%
result = {}
# example.features.feature is the dictionary
for key, feature in example.features.feature.items():
  # The values are the Feature objects which contain a `kind` which contains:
  # one of three fields: bytes_list, float_list, int64_list

  kind = feature.WhichOneof('kind')
  result[key] = np.array(getattr(feature, kind).value)

result

# %%
result['labels'].shape

# %%
result['points'].shape

# %% [markdown]
# ## Load dataset

# %%
PCLOUD_DIR = DATA_DIR / "pcloud"
EX_PCLOUD_PATH = PCLOUD_DIR / "00000002_pcloud_points.parq"


# %%
def _load_single_pcloud(path, label_type=""):
    label_col = f"is_{label_type}"
    points = (
        pd.read_parquet(path)
            .drop_duplicates(["x", "y", "z"])  # TODO shouuld keep the closest
            .assign(label=lambda df: df[label_col].astype(int))
    )
    point_coords = points[["x", "y", "z"]].values
    point_labels = points[["label"]].values
    return point_coords, point_labels


# %%
point_coords, point_labels = _load_single_pcloud(EX_PCLOUD_PATH, label_type="edge")


# %%
def load_dataset_from_dir(path, label_type="edge", n_files=None):
    paths = sorted(str(p) for p in path.glob("*.parq"))
    if n_files:
        paths = paths[:n_files]
    
    features, labels = [], []
    for path in paths:
        coords, per_pt_labels = _load_single_pcloud(path, label_type=label_type)
        features.append(coords)
        labels.append(per_pt_labels)
    
    return pd.DataFrame(dict(features=features, labels=labels))



# %%
dataset = load_dataset_from_dir(PCLOUD_DIR, n_files=100)
y = dataset.pop("labels")
x = dataset

# %%
x.features.values[0].shape

# %%
y.values[0].shape

# %% [markdown]
# # Misc

# %%
