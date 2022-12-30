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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# !python -V

# %% [markdown]
# ## Some example shapes in ABC dataset

# %%
# !ls 

# %%
STEP_CHUNK_PATH = "https://archive.nyu.edu/rest/bitstreams/88598/retrieve abc_0000_step_v00.7z"
OBJ_CHUCK_PATH = "https://archive.nyu.edu/rest/bitstreams/89085/retrieve abc_0000_obj_v00.7z"
FEAT_CHUNK_PATH = "https://archive.nyu.edu/rest/bitstreams/89087/retrieve abc_0000_feat_v00.7z"

# %%
EX_STEP_PATH = "data/00000050_80d90bfdd2e74e709956122a_step_000.step"
EX_OBJ_PATH = "data/00000050_80d90bfdd2e74e709956122a_trimesh_000.obj"
EX_FEAT_PATH = "data/00000050_80d90bfdd2e74e709956122a_features_000.yml"


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
# - [ ] figure out how to define ground truth from obj and/or feat.yaml file
#   - YL: 'Nearest neighbour' in PIE paper probably meant using the vertices defined in feature file and find the 1-NN

# %%
# !pip freeze | egrep 'pymesh'

# %%
# !pip install pymeshlab
# # !pip install polyscope  # optional to render the MeshSet state

# %%
def describe_mesh(mesh):
    print("# faces", mesh.face_number())
    print("# edges", mesh.edge_number())
    print("# vertices", mesh.vertex_number())


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
len(orig_points)

# %%
import random

SAMPLE_K = 8096

sampled_orig_points = np.array(random.sample(list(orig_points), SAMPLE_K))
sampled_orig_points[:3]


# %%
def plot_point_cloud(matrix):
    xs = matrix[:, 0]
    ys = matrix[:, 1]
    zs = matrix[:, 2]
    
    __, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw=dict(projection="3d"))
    ax.scatter(xs, ys, zs, s=1)
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))


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

# %%

# %%

# %%

# %% [markdown]
# ## Download Data
# #### `.obj` files

# %%
# !mkdir -p data/obj
# only the first chunk
# !head -n1 data/obj_v00.txt | xargs -n 2 -P 8 sh -c 'wget --no-check-certificate $0 -O data/obj/$1'

# %% [markdown]
# #### Corresponding feat files

# %%
# !mkdir -p data/feat
# only the first chunk
# !head -n1 data/feat_v00.txt | xargs -n 2 -P 8 sh -c 'wget --no-check-certificate $0 -O data/feat/$1'

# %%

# %%

# %% [markdown]
# ## MISC

# %% [markdown]
# ## An algorithm for sample points from CAD
# ![image.png](attachment:image.png)
# ref: https://www.researchgate.net/publication/308814491_3D_Modeling_by_Scanning_Physical_Modifications
