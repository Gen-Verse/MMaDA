# coding=utf-8
# Copyright 2025 MMaDA Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import wandb
from models import MAGVITv2, MMadaConfig, MMadaModelLM
from training.prompting_utils import UniversalPrompting
from training.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer, AutoConfig

def resize_vocab(model, config):
    print(f"Resizing token embeddings to {config.new_vocab_size}")
    model.resize_token_embeddings(config.new_vocab_size)

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

if __name__ == '__main__':

    config = get_config()
    resume_wandb_run = config.wandb.resume
    run_id = config.wandb.get("run_id", None)
    if run_id is None:
        resume_wandb_run = False
        run_id = wandb.util.generate_id()
        config.wandb.run_id = run_id

    wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}

    wandb.init(
        project="demo",
        name=config.experiment.name + '_mmu',
        config=wandb_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.mmada.pretrained_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob, use_reserved_token=True)

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()
    
    model = MMadaModelLM.from_pretrained(config.model.mmada.pretrained_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to(device)

    mask_token_id = model.config.mask_token_id

    temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 1  # retain only the top_k most likely tokens, clamp others to have 0 probability
    file_list = [f for f in os.listdir(config.mmu_image_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    responses = ['' for i in range(len(file_list))]
    images = []
    config.question = config.question.split(' *** ')
    for i, file_name in enumerate(tqdm(file_list)):
        image_path = os.path.join(config.mmu_image_root, file_name)
        image_ori = Image.open(image_path).convert("RGB")
        image = image_transform(image_ori, resolution=config.dataset.params.resolution).to(device)
        image = image.unsqueeze(0)
        images.append(image)
        image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
        batch_size = 1

        for question in config.question:
            input_ids = uni_prompting.text_tokenizer(['<|start_header_id|>user<|end_header_id|>\n' + "Please describe this image in detail."  +'<eot_id><|start_header_id|>assistant<|end_header_id|>\n'])['input_ids']
            input_ids = torch.tensor(input_ids).to(device)

            input_ids = torch.cat([
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
                image_tokens,
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
                input_ids
            ], dim=1).long()
            output_ids = model.mmu_generate(input_ids, max_new_tokens=1024, steps=512, block_length=1024)
            text = uni_prompting.text_tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
            print(text)
            responses[i] += f'User: ' + question + f'\n Answer : ' + text[0] + '\n'

    images = torch.cat(images, dim=0)
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    wandb_images = [wandb.Image(image, caption=responses[i]) for i, image in enumerate(pil_images)]
    wandb.log({"multimodal understanding": wandb_images}, step=0)

