import sys
from tqdm import tqdm
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    AutoModelForCausalLM,
)
sys.path.append('/aifs4su/yaodong/changye/TransformerLens')
from transformer_lens.HookedLlava import HookedLlava
import pdb
pdb.set_trace()
MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"

def load_models_and_processor(model_path):
    """
    加载处理器、视觉-语言模型和HookedTransformer语言模型。
    """
    # 加载处理器和视觉-语言模型
    processor = LlavaNextProcessor.from_pretrained(model_path)
    vision_model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.float32, 
        low_cpu_mem_usage=True,
        do_sample=False,
    )
    print("Vision model loaded.")
    
    # 加载 HookedTransformer 语言模型
    # hook_language_model = HookedLlava.from_pretrained(
    #     model_path,
    #     hf_model=vision_model.language_model,
    #     vision_tower=vision_model.vision_tower,
    #     multi_modal_projector=vision_model.multi_modal_projector,
    #     device="cuda", 
    #     fold_ln=False,
    #     center_writing_weights=False,
    #     center_unembed=False,
    #     tokenizer=None,
    #     dtype=torch.float32,
    # )
    # print(hook_language_model.state_dict().keys())
    # print(vision_model.language_model.state_dict().keys())
    # 将模型转移到GPU（如果可用）
    # hook_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vision_device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    # hook_language_model = hook_language_model.to(hook_device)
    vision_model = vision_model.to(vision_device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return processor, vision_model,  tokenizer

def consistent_check(model, hf_model, tokenizer):
    """
    检查 HookedTransformer 模型输出与 Hugging Face 模型输出的一致性。
    避免内存不够时的情况，逐步处理每个输入。
    """
    prompts = [
        "The capital of Germany is",
        "2 * 42 = ", 
        "My favorite", 
        "aosetuhaosuh aostud aoestuaoentsudhasuh aos tasat naostutshaosuhtnaoe usaho uaotsnhuaosntuhaosntu haouaoshat u saotheu saonuh aoesntuhaosut aosu thaosu thaoustaho usaothusaothuao sutao sutaotduaoetudet uaosthuao uaostuaoeu aostouhsaonh aosnthuaoscnuhaoshkbaoesnit haosuhaoe uasotehusntaosn.p.uo ksoentudhao ustahoeuaso usant.hsa otuhaotsi aostuhs",
    ]
    
    # 切换到评估模式
    model.eval()
    hf_model.eval()

    # 将模型参数移动到 GPU 上
    model_device = next(model.parameters()).device
    hf_model_device = next(hf_model.parameters()).device
    
    # 分别处理每一个 prompt，避免一次性加载太多
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}")

        # 将输入移动到模型所在的设备
        prompt_id = tokenizer.encode(prompt, return_tensors="pt").to(model_device)
        prompt_id_hf = tokenizer.encode(prompt, return_tensors="pt").to(hf_model_device)

        # 分别计算 HookedTransformer 和 Hugging Face 模型的输出
        tl_logits = model(prompt_id).detach().cpu()
        hf_logits = hf_model(prompt_id_hf).logits.detach().cpu()

        # 比较输出是否在允许的误差范围内
        if not torch.allclose(hf_logits, tl_logits, atol=1e-4, rtol=1e-2):
            print(f"Difference found in prompt {i}:")
            print(f"hf_logits: {hf_logits}")
            print(f"tl_logits: {tl_logits}")
            print(f"Difference: {hf_logits - tl_logits}")
            
            # 打印最大绝对误差和相对误差
            abs_diff = torch.max(torch.abs(hf_logits - tl_logits))
            rel_diff = torch.max(torch.abs((hf_logits - tl_logits) / (tl_logits + 1e-8)))
            print(f"Max absolute difference: {abs_diff.item()}")
            print(f"Max relative difference: {rel_diff.item()}")

            # 放宽误差条件
            if not torch.allclose(hf_logits, tl_logits, atol=1e-3, rtol=1e-2):
                print(f"Larger difference persists for prompt {i}, investigate further.")
        
        # 断言条件，严格验证差异
        assert torch.allclose(hf_logits, tl_logits, atol=1e-4, rtol=1e-2)

    print("Consistency check completed.")



def process_image_and_generate_response(processor, vision_model, image_path):
    """
    加载图像并生成图像描述。
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
    
    # 处理图像和文本输入
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:4")
    
    return inputs

def main():
    # 加载模型和处理器
    processor, vision_model, tokenizer= load_models_and_processor(MODEL_PATH)
    
    # 进行一致性检查
    
    # consistent_check(hook_language_model, vision_model.language_model, tokenizer)
    # 加载图像并生成响应
    image_path = "/aifs4su/yaodong/changye/TransformerLens/IMG_20230213_181559.jpg"
    inputs = process_image_and_generate_response(processor, vision_model, image_path)
    # outputs = hook_language_model.generate(inputs)
    # import pdb
    # pdb.set_trace()
    output = vision_model.generate(**inputs,max_new_tokens=100)
    
    print(processor.decode(output[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()