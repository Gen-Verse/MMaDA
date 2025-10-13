import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

from models import MAGVITv2, MMadaModelLM
from training.prompting_utils import UniversalPrompting
from training.utils import get_config, flatten_omega_conf, image_transform, image_transform_squash
from transformers import AutoTokenizer


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")


def _load_validation_data(prompts_file):
    with open(prompts_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("mmu_prompts_file must be a JSON list.")
    for i, item in enumerate(data):
        if "file_name" not in item or "messages" not in item:
            raise ValueError(f"Item {i} must contain 'file_name' and 'messages'.")
        if not isinstance(item["messages"], list):
            raise ValueError(f"Item {i} 'messages' must be a list.")
    return data


if __name__ == '__main__':
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mmu_model_path = config.get("mmu_model_path", None)
    if mmu_model_path is None:
        mmu_model_path = config.model.mmada.pretrained_model_path
    tokenizer = AutoTokenizer.from_pretrained(mmu_model_path, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=(
            "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
            "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
        ),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
        use_reserved_token=True
    )
    vq_cls = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_cls.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.eval()
    vq_model.requires_grad_(False)
    model = MMadaModelLM.from_pretrained(mmu_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    image_root = config.get("mmu_image_root", None)
    if image_root is None:
        image_root = getattr(config.dataset.params, "mmu_image_root", None)
    if image_root is None:
        raise ValueError("Please set mmu_image_root=... (or dataset.params.mmu_image_root).")
    prompts_file = config.get("mmu_prompts_file", None)
    if prompts_file is None:
        prompts_file = getattr(config.dataset.params, "mmu_validation_prompts_file", None)
    single_question = config.get("question", None)
    if prompts_file is None and single_question is None:
        raise ValueError("Please provide mmu_prompts_file=... (recommended) or question='...'(fallback).")
    if prompts_file is not None:
        validation_data = _load_validation_data(prompts_file)
    else:
        files = os.listdir(image_root)
        files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        validation_data = [
            {"file_name": f, "messages": [{"role": "user", "content": single_question}]}
            for f in sorted(files)
        ]
    results = []
    for item in tqdm(validation_data):
        file_name = item["file_name"]
        messages = item["messages"]
        image_path = os.path.join(image_root, file_name)
        if not os.path.exists(image_path):
            print(f"[Warn] Image not found: {image_path}. Skipped.")
            continue
        try:
            image_ori = Image.open(image_path).convert("RGB")
            if any(tag in file_name for tag in ['ai2d', 'clevr', 'docvqa', 'geo', 'llava']):
                image = image_transform_squash(image_ori, resolution=config.dataset.preprocessing.resolution).to(device)
            else:
                image = image_transform(image_ori, resolution=config.dataset.preprocessing.resolution).to(device)
            image = image.unsqueeze(0)
            image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
            text_token_ids = uni_prompting.text_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
            batch_size = image_tokens.shape[0]
            input_ids = torch.cat([
                (torch.ones(batch_size, 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
                (torch.ones(batch_size, 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
                image_tokens,
                (torch.ones(batch_size, 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
                text_token_ids
            ], dim=1).long()
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output_ids = model.mmu_generate(
                        input_ids,
                        max_new_tokens=config.dataset.preprocessing.max_seq_length,
                        steps=max(1, config.dataset.preprocessing.max_seq_length // 2),
                        block_length=max(1, config.dataset.preprocessing.max_seq_length // 4),
                    )
            else:
                output_ids = model.mmu_generate(
                    input_ids,
                    max_new_tokens=config.dataset.preprocessing.max_seq_length,
                    steps=max(1, config.dataset.preprocessing.max_seq_length // 2),
                    block_length=max(1, config.dataset.preprocessing.max_seq_length // 4),
                )
            generated_ids = output_ids[:, input_ids.shape[1]:]
            response_text = uni_prompting.text_tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            conversation_str = f"Image: {file_name}\n" + "=" * 20 + "\n"
            for msg in messages:
                role_prefix = "User: " if msg.get('role') == 'user' else "Assistant: "
                conversation_str += f"{role_prefix}{msg.get('content', '')}\n"
            conversation_str += f"Assistant (Generated): {response_text}\n"
            vis_img = torch.clamp((image.squeeze(0) + 1.0) / 2.0, min=0.0, max=1.0) * 255.0
            vis_img = vis_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            results.append({
                "file_name": file_name,
                "response": response_text,
                "conversation": conversation_str
            })
            print(f"[{file_name}] {response_text}")
        except Exception as e:
            print(f"[Error] {file_name}: {e}")
    out_path = os.path.join(os.getcwd(), "inference_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} results to {out_path}")
