# PIE-NET
(Re-)Implementation for PIE-NET paper [(Wang, et al., 2020)](https://arxiv.org/abs/2007.04883).

So far, training data preparation have been completed, and point classification is in a working state.
It is undecided whether curve proposal and proposal selection will be implemented.

## Methods
### Training Data
*For my quick study on point cloud sampling methods, see [this Markdown file](sampling_discussion.md).*

CAD models (`feat`s and `obj`s) were downloaded from [[2](https://deep-geometry.github.io/abc-dataset/)], before a Python package [[3](https://pymeshlab.readthedocs.io/en/latest/filter_list.html#generate_sampling_poisson_disk)] was used to sample point clouds from the `obj`s.
I experimented two sampling methods: Monte Carlo and Poisson Disk. The former was selected due to its faster speed of sampling, although the latter yielded better point clouds. For comparison, see [here](sampling_discussion.md).

Ground truth, including point classes and offsets, was calculated and transferred from the `feat`s to the sampled point clouds.

### Point Classification
In the paper, point classification was done with PointNet++. In this implementation, PointNet++ Tensorflow 2.0 layers were used from a public repo [[4](https://github.com/dgriffiths3/pointnet2-tensorflow2)].
I modified the existing semantic segmentation model to support point classification by adjusting final outputs and adding a custom loss.


## Failed Attempts
Some failed attempts are documented here as warnings for others to avoid.
- The original PointNet++ implementation [[5](https://github.com/charlesq34/pointnet2)] was not as useful because it requires older versions of Python and TF.
- Software like Blender or gmsh could also be used to sample point clouds from CAD models. But they generally do not have great Python APIs and come with redundant functionalities.


## Repo Structure
`training_data.py` includes some functions for generating training data.

The Pointnet++ layers were included as a submodule of this repo (`pointnet2-tensorflow2/`), which contains WIP point classification code.

`whiteboard.ipynb` is my interactive playground for testing and learning.


## References
1. [PIE-NET: Parametric Inference of Point Cloud Edges](https://arxiv.org/abs/2007.04883)
    - with [my notes](https://drive.google.com/file/d/1uBH-LoNl0S1QoQEb7XAt3GDNwUbJQQ8i/view?usp=sharing)
2. [ABC: A Big CAD Model Dataset For Geometric Deep Learning](https://deep-geometry.github.io/abc-dataset/)
3. [PyMeshLab](https://pymeshlab.readthedocs.io/en/latest/filter_list.html#generate_sampling_poisson_disk)
4. [Pointnet++ tensorflow 2.0 layers](https://github.com/dgriffiths3/pointnet2-tensorflow2)
5. [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://github.com/charlesq34/pointnet2)
