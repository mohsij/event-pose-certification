### Prerequisites
This repo has been tested with pytorch3d built from source using cuda 12.1
Therefore, please install cuda 12.1 according to official nvidia instructions or update your builds accordingly
cuda compiler version should be the same as the cuda version pytorch was built with essentially. Version mismatch will likely give you errors.

### Data
Download extras from the following link and merge with the repo directory
`https://drive.google.com/file/d/1NBFPczY8B5_RPXeLLID_eGEboB9xChq2/view?usp=drive_link`
*Note that the data is an improved version of the one in the paper and includes much more challenging conditions in some scenes along with a more realistic satellite model.

### Pytorch
1. Setup conda environment with python==3.9.18 (as tested)
2. Install pytorch `pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121`

### Pytorch3D (needed for chamfer loss)
1. Install `conda install -c iopath iopath`
2. `cd src/pytorch3d && pip install -e .` 

### Other setup
1. `pip install Cython yacs opencv-python==4.10.0.84 kornia==0.7.1 json-tricks scipy==1.13.1`
2. `cd src`
3. `cd lib`
4. `make`
5. `mkdir results`
6. `mkdir results/test-certifier/`
7. `mkdir log`
8. `mkdir output`
9. `<put pretrained models in the models directory>`
10. `export LD_LIBRARY_PATH=/home/<user-name>/anaconda3/envs/<conda-environment-name>/lib/:$LD_LIBRARY_PATH`
11. `sudo apt-get install libgeos-dev`

### Pre-training on synthetic
`python tools/train-on-synthetic.py --cfg experiments/train-synthetic-nocertifier.yaml`

### Self-supervised refinement
`python tools/train-selfsupervised-certifier.py --cfg experiments/test/ambient-sidetilt-constant.yaml`
`python tools/train-selfsupervised-certifier.py --cfg experiments/test/dark-sidetilt-constant.yaml`
`python tools/train-selfsupervised-certifier.py --cfg experiments/test/sun-sidetilt-constant.yaml`
