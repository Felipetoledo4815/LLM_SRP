from pathlib import Path

from vlm.SpaceLlavaWrapper import SpaceLlavaWrapper


def get_model():
    model_name = 'spacellava'
    vlm = SpaceLlavaWrapper(clip_path="./models/spacellava/mmproj-model-f16.gguf",
                                model_path="./models/spacellava/ggml-model-q4_0.gguf")
    return vlm
def main():
    model = get_model()
    prompt = Path('/home/radowanredoy/Desktop/rotation1/LLM_SRP/vlm/prompt/prompt_2.txt').read_text()
    llm_output, time_spent = model.inference(prompt, ['/home/radowanredoy/Desktop/rotation1/LLM_SRP/vlm/40.png'])
    print(llm_output)
    return

if __name__ == '__main__':
    main()