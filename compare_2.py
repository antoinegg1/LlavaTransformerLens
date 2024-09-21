import sys
from tqdm import tqdm
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)
sys.path.append('/aifs4su/yaodong/changye/TransformerLens')
from transformer_lens.HookedLlava import HookedLlava
import pdb
pdb.set_trace()
MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"

def load_models_and_processor(model_path):
    """
    加载处理器、视觉-语言模型和HookedTransformer语言模型，确保模型分配到不同的设备以避免显存不足。
    """
    # 加载处理器和视觉-语言模型
    processor = LlavaNextProcessor.from_pretrained(model_path)
    vision_model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.float32, 
        low_cpu_mem_usage=True
    )
    
    # 将 vision_tower 和 multi_modal_projector 分配到 cuda:0
    vision_tower = vision_model.vision_tower.to("cuda:0")
    multi_modal_projector = vision_model.multi_modal_projector.to("cuda:0")
    
    # HookedTransformer 语言模型分配到 cuda:1
    hook_language_model = HookedLlava.from_pretrained(
        model_path,
        hf_model=vision_model.language_model,
        vision_tower=vision_tower,
        multi_modal_projector=multi_modal_projector,
        device="cuda:1",  # 放在cuda:1
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        do_sample=False,
        tokenizer=None,
        dtype=torch.float32,
    )
    
    # vision_model 也放置到 cuda:0
    vision_model = vision_model.to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return processor, vision_model, tokenizer, hook_language_model

def process_image_and_generate_response(processor, vision_model, image_path, device_vision="cuda:0"):
    """
    加载图像并生成图像描述，处理时确保数据放置在vision模型所在的设备。
    """
    # 加载本地图像
    image = Image.open(image_path)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown in this image?"},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # 处理图像和文本输入，确保数据在 vision 模型所在设备上
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device_vision)
    
    return inputs

def main():
    # 加载模型和处理器
    processor, vision_model, tokenizer, hook_language_model = load_models_and_processor(MODEL_PATH)
    
    # 加载图像并生成响应
    image_path = "/aifs4su/yaodong/changye/TransformerLens/IMG_20230213_181559.jpg"
    
    # 处理图像输入在 cuda:0 上
    inputs = process_image_and_generate_response(processor, vision_model, image_path, device_vision="cuda:0")
    
    # 将输入从 cuda:0 移动到 hook_language_model 所在的 cuda:1 设备
    inputs = inputs.to("cuda:1")
    
    # 在 hook_language_model 上生成输出
    outputs = hook_language_model.generate(inputs)
    
    # 解码并打印输出
    print(processor.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
