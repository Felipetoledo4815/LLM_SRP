# LLM_SRP

## Get started
1. Download Nuscenes from https://www.nuscenes.org/nuscenes. You will need to create an account. To get started just download the mini partition and set it up by following: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/nuscenes_tutorial.ipynb.
2. Create a conda environment and install the requirements by running:
```bash
conda env create -n llm_srp
conda activate llm_srp
pip install -r requirements.txt
```
3. Create a `.env` file in the root directory and add the following according to what partition you want to use:
```
NUSCENES_MINI=/path/to/nuscenes_mini
NUSCENES=/path/to/nuscenes
```
4. Explore the dataset using the `explore_llm_srp.ipynb` notebook.