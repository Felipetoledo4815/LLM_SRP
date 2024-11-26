# Download Models

## SpaceLlava
Create a folder `./models/spacellava` and download the following files inside of it:
- [ggml-model-q4_0.gguf](https://huggingface.co/remyxai/SpaceLLaVA/blob/main/ggml-model-q4_0.gguf)
- [mmproj-model-f16.gguf](https://huggingface.co/remyxai/SpaceLLaVA/blob/main/mmproj-model-f16.gguf)
Do not change the names of the files.

Note: to ensure that llama-cpp works on the GPU, you need to install llama-cpp-python with CUBLAS enabled. You can do this by executing:
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.2.45 --force-reinstall --no-cache-dir
```

## Llava 1.5 7b (Hugging Face)
This model should be downloaded automatically when executing the benchmark.

## Fine-tuned Llava 1.5 7b (LlaVA)
To load a custom Llava 1.5 model, we need to download the LlaVA repository. To do so, reate a folder `./external_code`, move to that folder, and then clone the repository inside of it:
```bash
mkdir -p external_code
cd external_code
git clone https://github.com/haotian-liu/LLaVA.git
cd LlaVA
git checkout c121f0432da27facab705978f83c4ada465e46fd
```

Then, open `./external_code/LlaVA/llava/model/language_model/llava_llama.py` and add `cache_position: None = None,` between line 65 and 66.

Finally, install LlaVA in your environment by executing:
```bash
pip install -e .
cd ../../
```

## Llava 1.6 Mistral 7b (Hugging Face)
This model should be downloaded automatically when executing the benchmark.

## Llava 1.6 Vicuna 7b (Hugging Face)
This model should be downloaded automatically when executing the benchmark.

## Cambrian Llama3-8 and Phi3-3b
TODO: Add instructions to download the models, and how to install library (Follow their readme instructions).
Problem: cambrian needs:
```
transformers==4.37.0
tokenizers==0.15.0
```
But those versions are incompatible with LlavaNext, which needs:
```
transformers==4.43.3
tokenizers==0.19.1
```

## RoadScene2Vec
This is a scene graph generator (SGG) from https://github.com/AICPS/roadscene2vec.
Create a folder `./external_code`, move to that folder, and then clone the repository inside of it:
```bash
mkdir -p external_code
cd external_code
git clone https://github.com/AICPS/roadscene2vec.git
```
After that you need to copy the `./models/roadscene2vec/setup.py` file to the `./external_code/roadscene2vec` folder.
```bash
cd ..
cp ./models/roadscene2vec/setup.py ./external_code/roadscene2vec
```
Open `./external_code/roadscene2vec/roadscene2vec/util/visualizer.py`, and replace line 19 `matplotlib.use('TkAgg')` with `matplotlib.use('Agg')`.

Then, open `./external_code/roadscene2vec/requirements.txt` and comment the line `scikit-learn~=0.24.2`.

Finally, make sure you have gcc installed (tested with gcc v-11.4.0) and run the following command to install the roadscene2vec package:
```bash
cd external_code/roadscene2vec
pip install -r requirements.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install -e .
cd ../../
```