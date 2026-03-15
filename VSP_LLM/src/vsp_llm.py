# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import sys, logging
import contextlib
from argparse import Namespace
from typing import Dict, List, Optional, Tuple, Any
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from einops import repeat

from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from omegaconf import II, MISSING

from transformers import Qwen3VLVisionModel,Qwen3VLForConditionalGeneration


logger = logging.getLogger(__name__)


MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(
    ["static", "uniform", "normal", "poisson"]
)


@dataclass
class VSPLLMConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to hubert model"}
    )
    llm_ckpt_path: str = field(
        default=MISSING, metadata={"help": "path to llama model"}
    )
    no_pretrained_weights: bool = field(
        default=False,
        metadata={"help": "if true, does not load pretrained weights"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout after transformer and before final projection"
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability inside hubert model"},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights "
                    "inside hubert model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
                    "inside hubert model"
        },
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask "
                    "(normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
                    "(used for more complex distributions), "
                    "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
                    "(used for more complex distributions), "
                    "see help in compute_mask_indices"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    masking_updates: int = field(
        default=0,
        metadata={"help": "dont finetune hubert for this many updates"},
    )
    feature_grad_mult: float = field(
        default=0.0,
        metadata={"help": "reset feature grad mult in hubert to this"},
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a layer in hubert"},
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")

    # this holds the loaded hubert args
    w2v_args: Any = None
    encoder_embed_dim: int = field(
        default=1024, metadata={"help": "encoder embedding dimension"}
    )
    decoder_embed_dim: int = field(
        default=4096, metadata={"help": "decoder embedding dimension"}
    )
    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "dont finetune hubert for this many updates"},
    )
    ft_ckpt_path: str = field(
        default=-1, metadata={"help": "path to checkpoint for finetuning"}
    )



class HubertEncoderWrapper(FairseqEncoder):
    def __init__(self, w2v_model):
        super().__init__(None)
        self.w2v_model = w2v_model

    def forward_(self, source, padding_mask, **kwargs):
        src ={}
        src['video'] = source
        src['audio'] = None
        w2v_args = {
            "source": src,
            "padding_mask": padding_mask,
        }

        x, padding_mask = self.w2v_model.extract_finetune(**w2v_args)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask
        }


    def forward(self, source, padding_mask, **kwargs):
            w2v_args = {
                "source": source,
                "padding_mask": padding_mask,
            }

            x, padding_mask = self.w2v_model.extract_finetune(**w2v_args)


            return {
                "encoder_out": x,  # T x B x C
                "encoder_padding_mask": padding_mask,  # B x T
                "padding_mask": padding_mask
            }
 

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out[
                "padding_mask"
            ].index_select(0, new_order)
        return encoder_out




@register_model("vsp_llm", dataclass=VSPLLMConfig)
class avhubert_llm_seq2seq_cluster_count(BaseFairseqModel):
    def __init__(self, encoder, decoder, cfg,version="qwen",video_encoder=None):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.decoder = decoder
        self.avfeat_to_llm = nn.Linear(1024, 4096)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.version = version
        self.video_encoder=video_encoder
        if version=="qwen":
            self.qwen_to_llm = nn.Linear(1152, 4096)
        
    @classmethod
    def build_model(cls, cfg, task,version="qwen"):
        """Build a new model instance."""

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }
        print(cfg)
        print("CFG ABOVE!")

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )
        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )
        w2v_args.task.data = cfg.data

        task_pretrain = tasks.setup_task(w2v_args.task)
        print(w2v_args.task)
        print("ABOVE1")
        
        if state is not None:
            task_pretrain.load_state_dict(state['task_state'])

        encoder_ = task_pretrain.build_model(w2v_args.model)

        encoder = HubertEncoderWrapper(encoder_)
        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            del state['model']['mask_emb']
            encoder.w2v_model.load_state_dict(state["model"], strict=False)

        encoder.w2v_model.remove_pretraining_modules()
        # print("CFG below:")
        # print(cfg) # no_pretrained_weights=False
        # print("w2v_args below:")
        # print(w2v_args)
        
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )

        # decoder_4bit = AutoModelForCausalLM.from_pretrained(cfg.llm_ckpt_path, quantization_config=bnb_config)           
        
        if version == "llama":
            decoder= AutoModelForCausalLM.from_pretrained(cfg.llm_ckpt_path) 


            config = LoraConfig(
                r=16, 
                lora_alpha=32, 
                target_modules=["q_proj", "v_proj", "k_proj"], 
                lora_dropout=0.05, 
                bias="none", 
                task_type="CAUSAL_LM" 
            )

            decoder = get_peft_model(decoder, config)
            decoder.print_trainable_parameters()
            
            model = avhubert_llm_seq2seq_cluster_count(encoder, decoder, cfg,version="llama")
            if cfg.ft_ckpt_path != "-1":
                state = checkpoint_utils.load_checkpoint_to_cpu(cfg.ft_ckpt_path)

                # Remove incompatible decoder token weights
                for key in list(state["model"].keys()):
                    if key in ["decoder.embed_tokens.weight", "decoder.embed_out"]:
                        state["model"].pop(key, None)

                model.load_state_dict(state["model"], strict=False)

                print("\nAVHubertSeq2Seq loaded from", cfg.ft_ckpt_path, "\n")

            return model
                
        elif version=="qwen": 
            # qwen3vl
            # video_encoder = Qwen3VLVisionModel.from_pretrained(
            #     "/mnt/sdb/yuran/av_hubert_llm/VSP-LLM/checkpoints/Qwen3-VL-8B-Instruct",
            # )
            whole_qwen3=Qwen3VLForConditionalGeneration.from_pretrained(
                "/mnt/sdb/yuran/av_hubert_llm/VSP-LLM/checkpoints/Qwen3-VL-8B-Instruct",
            )
            video_encoder = whole_qwen3.model.visual
            del whole_qwen3
            
            decoder= AutoModelForCausalLM.from_pretrained(cfg.llm_ckpt_path) 


            config = LoraConfig(
                r=16, 
                lora_alpha=32, 
                target_modules=["q_proj", "v_proj", "k_proj"], 
                lora_dropout=0.05, 
                bias="none", 
                task_type="CAUSAL_LM" 
            )

            decoder = get_peft_model(decoder, config)
            decoder.print_trainable_parameters()
            
            model = avhubert_llm_seq2seq_cluster_count(encoder, decoder, cfg,version="qwen",video_encoder=video_encoder)
            if cfg.ft_ckpt_path != "-1":
                state = checkpoint_utils.load_checkpoint_to_cpu(cfg.ft_ckpt_path)

                # Remove incompatible decoder token weights
                for key in list(state["model"].keys()):
                    if key in ["decoder.embed_tokens.weight", "decoder.embed_out"]:
                        state["model"].pop(key, None)

                model.load_state_dict(state["model"], strict=False)

                print("\nAVHubertSeq2Seq loaded from", cfg.ft_ckpt_path, "\n")

            return model
            
    
    def forward(self, **kwargs):
        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            output = self.encoder(**kwargs)
    
        output['encoder_out'] = self.avfeat_to_llm(output['encoder_out'])
        
        cluster_counts = kwargs['source']['cluster_counts'][0] # tensor list
        
        results_tensor = []
        start_idx = 0
        for clutser_num in cluster_counts:
            end_idx = start_idx + clutser_num
            slice = output['encoder_out'][:,start_idx:end_idx,:]
            mean_tensor = torch.mean(slice, dim=1, keepdim=True)
            results_tensor.append(mean_tensor)            
            start_idx = end_idx

        assert(cluster_counts.sum().item() == output['encoder_out'].size()[1])
        
        if self.version == "llama":
            # llama case, pure text model

            reduced_enc_out = torch.cat(results_tensor, dim=1)      
            B, T, D = reduced_enc_out.size()
            instruction = kwargs['source']['text']
            instruction_embedding = self.decoder.model.model.embed_tokens(instruction)

            labels = kwargs['target_list'].clone().to(device=self.decoder.device, dtype=torch.long)
            labels_embedding = self.decoder.model.model.embed_tokens(labels)

            llm_input = torch.cat((instruction_embedding, reduced_enc_out, labels_embedding), dim=1)
            llm_labels = labels.clone()
            llm_labels[llm_labels == 0] = -100
            
            _, instruction_embedding_t, _ = instruction_embedding.size()
            target_ids = torch.full((B, T + instruction_embedding_t),-100).long().to(labels.device)
            llm_labels = torch.cat((target_ids, llm_labels), dim=1)
            llm_out = self.decoder(inputs_embeds=llm_input, labels=llm_labels, return_dict=True)
            
            return llm_out.loss, llm_out.logits
        elif self.version == "qwen":
            # ================================
            # Qwen3-VL multimodal branch
            # ================================
            self.video_encoder = self.video_encoder
            
            reduced_enc_out = torch.cat(results_tensor, dim=1)
            B, T, D = reduced_enc_out.size()

            # input_ids = kwargs['source']["input_ids"][0].to(self.decoder.device)
            # attention_mask = kwargs['source']["attention_mask"][0].to(self.decoder.device)
            pixel_values_videos = kwargs['source']["pixel_values_videos"][0]
            video_grid_thw = kwargs['source']["video_grid_thw"][0]
            
            
            video_features = self.video_encoder(
                hidden_states=pixel_values_videos,
                grid_thw=video_grid_thw
            ).last_hidden_state /100  
            video_features=video_features.detach()# detach to have no grad to video encoder
            
            
            instruction = kwargs['source']['text']
            instruction_embedding = self.decoder.model.model.embed_tokens(instruction)

            #print("VIDEO FEATURES", video_features.shape) #960,1152
            proj_video_features=self.qwen_to_llm(video_features).unsqueeze(0) # 1,960,4096
            
            labels = kwargs['target_list'].clone().to(device=self.decoder.device, dtype=torch.long)
            labels_embedding = self.decoder.model.model.embed_tokens(labels)

            llm_input = torch.cat((instruction_embedding, reduced_enc_out,proj_video_features, labels_embedding), dim=1)
            llm_labels = labels.clone()
            llm_labels[llm_labels == 0] = -100
            
            _, instruction_embedding_t, _ = instruction_embedding.size()
            _, qwen_t, _ = proj_video_features.size()
            target_ids = torch.full((B, T + instruction_embedding_t + qwen_t),-100).long().to(labels.device)
            # print("hubert, instruction, video",T, instruction_embedding_t, qwen_t)
            llm_labels = torch.cat((target_ids, llm_labels), dim=1)
            llm_out = self.decoder(inputs_embeds=llm_input, labels=llm_labels, return_dict=True)
            
            return llm_out.loss, llm_out.logits
            # position_ids=kwargs["position_ids"]

            # labels = kwargs["source"]["labels_qwen"][0].clone().to(self.decoder.device)
            

            # ------------------------------------------------
            # 1. Build Qwen token embeddings
            # ------------------------------------------------
            # with torch.no_grad():
            #     text_embeds = self.decoder.model.get_input_embeddings()(input_ids)


            # avhubert_embeds = reduced_enc_out.to(text_embeds.dtype)

            # ------------------------------------------------
            # 3. Prepend AVHubert tokens
            # ------------------------------------------------

            # llm_input = torch.cat(
            #     (avhubert_embeds, text_embeds),
            #     dim=1
            # )

            # ------------------------------------------------
            # 4. Extend attention mask
            # ------------------------------------------------

            # hubert_mask = torch.ones(
            #     B,
            #     T_av,
            #     dtype=attention_mask.dtype,
            #     device=attention_mask.device
            # )

            # attention_mask = torch.cat(
            #     (hubert_mask, attention_mask),
            #     dim=1
            # )

            # ------------------------------------------------
            # 5. Extend labels (ignore AVHubert tokens)
            # ------------------------------------------------

            # labels[labels == 0] = -100

            # hubert_labels = torch.full(
            #     (B, T_av),
            #     -100,
            #     dtype=labels.dtype,
            #     device=labels.device
            # )

            # llm_labels = torch.cat(
            #     (hubert_labels, labels),
            #     dim=1
            # )

            # ------------------------------------------------
            # 6. Forward through Qwen3-VL
            # ------------------------------------------------

            # llm_out = self.decoder(
            #     inputs_embeds=llm_input,
            #     attention_mask=attention_mask,
            #     #pixel_values_videos=pixel_values_videos,
            #     #video_grid_thw=video_grid_thw,
            #     labels=llm_labels,
            #     return_dict=True
            #     # FIXME must pass position ids!
            # )

            # return llm_out.loss, llm_out.logits,llm_labels,T_av
        



    @torch.no_grad()
    def generate(self,
                num_beams=20,
                max_length=30,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.0,
                length_penalty=0.0,
                  **kwargs,
                ):
        output = self.encoder(**kwargs)
        output['encoder_out'] = self.avfeat_to_llm(output['encoder_out'])
        cluster_counts = kwargs['source']['cluster_counts'][0] # tensor list
        
        results_tensor = []
        start_idx = 0

        for clutser_num in cluster_counts:
            end_idx = start_idx + clutser_num
            slice = output['encoder_out'][:,start_idx:end_idx,:]
            mean_tensor = torch.mean(slice, dim=1, keepdim=True)
            results_tensor.append(mean_tensor)            
            start_idx = end_idx

        assert(cluster_counts.sum().item() == output['encoder_out'].size()[1])

        reduced_enc_out = torch.cat(results_tensor, dim=1)     
        B, T, D = reduced_enc_out.size()
        
        if self.version == "llama":
            instruction = kwargs['source']['text'].to(device=self.decoder.device, dtype=torch.long)
            instruction_embedding = self.decoder.model.model.embed_tokens(instruction)
            llm_input = torch.cat((instruction_embedding, reduced_enc_out.to(self.decoder.device)), dim=1) 

            self.decoder.config.use_cache = True
            outputs = self.decoder.generate(inputs_embeds=llm_input,
                            top_p=top_p,
                            num_beams=num_beams,
                            max_new_tokens=max_length,
                            min_length=min_length,
                            repetition_penalty=repetition_penalty,
                            do_sample=True,
                            length_penalty=length_penalty,
                            )

            return outputs
        elif self.version == "qwen":
            # ================================
            # Qwen3-VL multimodal generation
            # ================================
            B, T, D = reduced_enc_out.size()

            # Video inputs
            pixel_values_videos = kwargs['source']["pixel_values_videos"][0].to(self.decoder.device)
            video_grid_thw = kwargs['source']["video_grid_thw"][0]

            # Compute video features via video encoder
            video_features = self.video_encoder(
                hidden_states=pixel_values_videos,
                grid_thw=video_grid_thw
            ).last_hidden_state / 100  # match forward() scaling

            # Project video features to LLM dimension
            proj_video_features = self.qwen_to_llm(video_features).unsqueeze(0)  # 1 x T_v x 4096

            # Instruction embeddings
            instruction = kwargs['source']['text'].to(self.decoder.device, dtype=torch.long)
            instruction_embedding = self.decoder.model.model.embed_tokens(instruction)

            # Concatenate embeddings: instruction + AVHubert + video
            llm_input = torch.cat((instruction_embedding, reduced_enc_out.to(self.decoder.device), proj_video_features), dim=1)

            # Set caching for generation
            self.decoder.config.use_cache = True

            # Generate
            outputs = self.decoder.generate(
                inputs_embeds=llm_input,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                length_penalty=length_penalty,
            )
            return outputs
        
    def get_ctc_target(self, sample):
        return sample["target"], sample["target_lengths"]

    def get_ctc_output(self, encoder_out, sample):
        en_out = encoder_out["encoder_out"]
        logits = self.ctc_proj(en_out)  # T x B x C
        out = utils.log_softmax(logits.float(), dim=-1)
        padding_mask = encoder_out["encoder_padding_mask"]
        lens = out.new_full((out.shape[1],), out.shape[0]).long()
        if len(padding_mask) > 0:
            lens -= padding_mask[0].sum(dim=-1)
        return out, lens

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def state_dict(self):
        old_state = super().state_dict()
        state = {k:v for k,v in old_state.items() if 'lora' in k or 'avfeat_to_llm' in k or 'encoder' in k or "qwen" in k}
        return state


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
