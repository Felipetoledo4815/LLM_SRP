# LLM_SRP

## Getting started

1. Create a local `.env` file on the root of the repository containing the following variables:
```bash
NUSCENES_MINI=/path/to/nuscenes_mini
NUSCENES=/path/to/nuscenes
```
2. Create a conda environment with the following command:
```bash
conda create -n llm_srp python=3.10
```
3. Activate the conda environment:
```bash
conda activate llm_srp
```
4. Install all packages in the requirements.txt file:
```bash
pip install -r requirements.txt
```

## Exploring the dataset
- If you want to have a quick overview of the dataset, you can use the jupyter-notebook called `explore_llm_srp.ipynb`.
- To execute minimal versions of different scripts in this repository, execute the following command from the root of the repository:
```bash
python -m dataset.utils.tests.run_scene_plot
```

## Installing the package
If you want to install the package to use it in other projects, you can run the following command from the root of the repository:
```bash
pip install -e .
```