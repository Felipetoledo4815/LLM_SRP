# Monitor

## Getting started

1. You need to have installed the llm_srp conda environment. For that follow the instruction in [README](../README.md).
2. Once installed, activate it by running:
```bash
conda activate .llm_srp/
```
3. Move to the monitor folder and execute `dowloader.py` to obtain the VLM's predictions and test set labels:
```bash
cd monitor/
python downloader.py
```
4. To execute the monitor execute:
```bash
python run.py
```
5. To analyze the results you can execute the python notebook `./monitor/analyze_results.ipynb`.
**NOTE** that you will not be able to print the waymo images in the last section unless you have downloaded the waymo dataset. 
Information about how to download it can be found at [README](../dataset/README.md).