from safetensors.torch import load_file
from safetensors.torch import save_file
import torch # 尽管代码里没直接用，但 safetensors 加载时需要它

file_path = ".cache/bitnet-b1.58-2B-4T/new_model.safetensors"


model_dict = load_file(file_path)


for key, value in model_dict.items():
    print(f"名称为{key},类型为{value.dtype},形状为{value.shape} \n")