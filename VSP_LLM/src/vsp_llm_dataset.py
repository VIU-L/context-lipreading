# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import time
from typing import Any, List, Optional, Union
import json
from tqdm import tqdm
import random
# from rope2d import get_rope_index_2

import numpy as np

import sys
sys.path.append("/mnt/sdb/yuran/av_hubert_llm/Qwen3-VL/qwen-vl-utils/src")
from qwen_vl_utils import process_vision_info

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from python_speech_features import logfbank
from scipy.io import wavfile
from transformers import AutoTokenizer, AutoProcessor
DBG=True if len(sys.argv) == 1 else False

if DBG:
    import utils as custom_utils
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "DEBUG").upper(),
        stream=sys.stdout,
    )
else:
    from . import utils_vsp_llm as custom_utils

logger = logging.getLogger(__name__)


def load_audio_visual(manifest_path, max_keep, min_keep, frame_rate, label_paths, label_rates, tol=0.1):
    def is_audio_label_aligned(audio_dur, label_durs):
        return all([abs(audio_dur - label_dur)<tol for label_dur in label_durs])

    n_long, n_short, n_unaligned = 0, 0, 0
    names, inds, sizes = [], [], []
    dur_from_label_list = []
    is_seq_label = any([x==-1 for x in label_rates])
    for label_path, label_rate in zip(label_paths, label_rates):
        label_lengths = [len(line.rstrip().split())/label_rate for line in open(label_path).readlines()]
        dur_from_label_list.append(label_lengths)
    dur_from_label_list = list(zip(*dur_from_label_list))
    
    manifest = manifest_path.split('/')[-1].split('.')[0]
    cluster_counts_fn = manifest_path.replace('.tsv','.cluster_counts')
    cluster_counts_list = open(cluster_counts_fn).readlines()
    
    
    cluster_counts = []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            sz = int(items[-2]) # 
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            elif (not is_seq_label) and (not is_audio_label_aligned(sz/frame_rate, dur_from_label_list[ind])):
                n_unaligned += 1
            else:
                video_path = items[1]
                audio_path = items[2]
                audio_id = items[0]
                names.append((video_path, audio_path+':'+audio_id))
                inds.append(ind)
                sizes.append(sz)
                cluster_counts.append(cluster_counts_list[ind].strip())
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long and {n_unaligned} unaligned, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, inds, tot, sizes, cluster_counts

def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    with open(label_path) as f:
        lengths = [len(line.rstrip().split()) for line in f]
        assert len(lengths) == tot
        lengths = [lengths[i] for i in inds]
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate
        dur_from_label = lengths[i] / label_rate
        if abs(dur_from_audio - dur_from_label) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )


class VSP_LLM_dataset(FairseqDataset):
    def __init__(
            self,
            manifest_path: str,
            sample_rate: float,
            llm_ckpt_path: str,
            label_paths: List[str],
            label_rates: Union[List[float], float],  # -1 for sequence labels
            max_keep_sample_size: Optional[int] = None,
            min_keep_sample_size: Optional[int] = None,
            max_sample_size: Optional[int] = None,
            shuffle: bool = True,
            pad_audio: bool = False,
            normalize: bool = False,
            store_labels: bool = True,
            random_crop: bool = False,
            single_target: bool = False,
            stack_order_audio: int=1,
            skip_verify: bool=False,
            image_mean: float=0,
            image_std: float=1,
            image_crop_size: int=88,
            image_aug: bool=False,
            modalities: Optional[List[str]]=None,
            is_s2s=False,
            noise_fn=None,
            noise_prob=0,
            noise_snr=0,
            noise_num=1,
            use_topic_sentence_rate=0,
            use_keyword_rate=0,
            use_voxmm_topic_sentence_rate=0,
            use_uncropped_video=0,
            use_qwen_topic_sentence_rate=0,
            use_multivsr_topic_sentence_rate=0
    ):
        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, int)
            else label_rates
        )
        self.modalities = set(modalities)
        self.audio_root, self.names, inds, tot, self.sizes, self.cluster_counts, = load_audio_visual(manifest_path, max_keep_sample_size, min_keep_sample_size, frame_rate=sample_rate, label_paths=label_paths, label_rates=self.label_rates)
        self.sample_rate = sample_rate
        self.stack_order_audio = stack_order_audio
        self.shuffle = shuffle
        self.random_crop = random_crop
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_ckpt_path) # this is auto adapted to different llm.
        if use_uncropped_video>0:
            self.qwen_processor = AutoProcessor.from_pretrained("/mnt/sdb/yuran/av_hubert_llm/VSP-LLM/checkpoints/Qwen3-VL-8B-Instruct")
        self.num_labels = len(label_paths)
        self.single_target = single_target
        self.store_labels = store_labels
        self.is_s2s = is_s2s
        self.noise_wav, self.noise_prob, self.noise_snr, self.noise_num = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else [], noise_prob, noise_snr, noise_num
        self.lang_dict = {'en':"English", 'es':"Spanish", 'fr':"French", 'it':'Italian', 'pt':"Portuguese"}
        self.use_uncropped_video=use_uncropped_video

        assert self.single_target == (self.label_rates[0] == -1), f"single target should be equivalent to sequence label (label_rate==-1)"
        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]
        if not skip_verify:
            for label_path, label_rate in zip(label_paths, self.label_rates):
                verify_label_lengths(self.sizes, self.sample_rate, label_path, label_rate, inds, tot)
        else:
            logger.info(f"Skip label alignment verifying")

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        if image_aug:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.RandomCrop((image_crop_size, image_crop_size)),
                custom_utils.HorizontalFlip(0.5),
                custom_utils.Normalize(image_mean, image_std) ])
        else:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.CenterCrop((image_crop_size, image_crop_size)),
                custom_utils.Normalize(image_mean, image_std) ])
        logger.info(f"image transform: {self.transform}")

        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}, "
            f"seqs2seq data={self.is_s2s},")
        logger.info(
            f"Noise wav: {noise_fn}->{len(self.noise_wav)} wav, Prob: {self.noise_prob}, SNR: {self.noise_snr}, Number of mixture: {self.noise_num}"
        )
        self.use_topic_sentence_rate=use_topic_sentence_rate
        if use_topic_sentence_rate>0:
            
            self.topic_path="/mnt/sdb/yuran/av_hubert/datasets/lrs2/topics/segments_with_topic_fixed.jsonl"
            # one line:
            # {"line_idx": 0, "split": "pretrain", "video_id": "5535415699068794046", "segment_id": "00001", "segment_id_int": 1, "rel_path": "pretrain/5535415699068794046/00001", "transcript": "...", "n_words": 29, "keywords_5": ["cooking", "ingredients", "food preparation", "texture", "culinary techniques"], "topic_sentence": "Exploring methods and characteristics involved in preparing and evaluating various foods at home."}
            self.rel_path_to_topic_sentence={}
            with open(self.topic_path,'r') as file:
                for line in tqdm(file,desc="loading topics"):
                    line_dict=json.loads(line)
                    self.rel_path_to_topic_sentence[line_dict["rel_path"]]=line_dict["topic_sentence"]
                    
        self.use_keyword_rate=use_keyword_rate
        if use_keyword_rate>0:
            self.topic_path="/mnt/sdb/yuran/av_hubert/datasets/lrs2/topics/segments_with_topic_fixed.jsonl"
            # one line:
            # {"line_idx": 0, "split": "pretrain", "video_id": "5535415699068794046", "segment_id": "00001", "segment_id_int": 1, "rel_path": "pretrain/5535415699068794046/00001", "transcript": "...", "n_words": 29, "keywords_5": ["cooking", "ingredients", "food preparation", "texture", "culinary techniques"], "topic_sentence": "Exploring methods and characteristics involved in preparing and evaluating various foods at home."}
            self.rel_path_to_keyword={}
            with open(self.topic_path,'r') as file:
                for line in tqdm(file,desc="loading topics"):
                    line_dict=json.loads(line)
                    k1, k2, k3, k4, k5 = random.sample(line_dict["keywords_5"], 5)           # permute
                             
                    self.rel_path_to_keyword[line_dict["rel_path"]]=k1+", "+k2+", "+k3+", "+k4+", "+k5+"."
        self.use_voxmm_topic_sentence_rate=use_voxmm_topic_sentence_rate
        if use_voxmm_topic_sentence_rate>0:
            self.voxmm_topic_path="/mnt/sdb/yuran/av_hubert/datasets/voxmm/raw/video_id_to_topic.json"
            # format: video_id : sentence
            with open(self.voxmm_topic_path,'r') as file:
                self.video_id_to_topic_sentence=json.load(file)
            print(f"Loaded voxmm video_id to topic sentence mapping, total {len(self.video_id_to_topic_sentence)} entries.")
        if use_uncropped_video>0:
            self.uncropped_video_root = "/mnt/sdb/yuran/av_hubert/datasets/multivsr/multivsr_nocrop"
            
            
        self.use_qwen_topic_sentence_rate=use_qwen_topic_sentence_rate
        if use_qwen_topic_sentence_rate>0:
            from pathlib import Path
            self.qwen_topic_sentence_root="/mnt/sdb/yuran/av_hubert/datasets/multivsr/multivsr_nocrop"
            # walk through root/video_id/segment_id.txt,collect "video_id/segment_id" : content str
            self.qwen_topic_sentence_dict = {}
            root=Path(self.qwen_topic_sentence_root)
            for txt_path in tqdm(root.rglob("*.txt"),desc="PROBING TXTs"):
                # txt_path: root/video_id/segment_id.txt
            
                video_id = txt_path.parent.name
                segment_id = txt_path.stem  # remove .txt
                if segment_id[-2:] != "v3": # only keep version 3
                    continue
                else:
                    segment_id=segment_id[:-2]
                
                key = f"{video_id}/{segment_id}"
                
                with open(txt_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                
                self.qwen_topic_sentence_dict[key] = content
        self.use_multivsr_topic_sentence_rate=use_multivsr_topic_sentence_rate
        if use_multivsr_topic_sentence_rate>0:
            multivsr_topic_sentence_path="/mnt/sdb/yuran/av_hubert/datasets/multivsr/ytb_metadata_purged.json"
            with open(multivsr_topic_sentence_path,'r') as f:
                self.multivsr_topic_sentence_dict=json.load(f)
            print("METADATA LOADED! length:"+str(len(self.multivsr_topic_sentence_dict)))
                
            
    def load_units(self, index):
        #assert('video' in self.modalities and 'audio' in self.modalities)
        av_units = self.cluster_counts[index].strip().split(' ')
        int_av_units = [int(x) for x in av_units]
        av_units = torch.tensor(int_av_units, dtype=int)

        return av_units
        
    def load_feature(self, mix_name):
        """
        Load image and audio feature
        Returns:
        video_feats: numpy.ndarray of shape [T, H, W, 1], audio_feats: numpy.ndarray of shape [T, F]
        """
        def stacker(feats, stack_order):
            """
            Concatenating consecutive audio frames
            Args:
            feats - numpy.ndarray of shape [T, F]
            stack_order - int (number of neighboring frames to concatenate
            Returns:
            feats - numpy.ndarray of shape [T', F']
            """
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
            return feats
        video_fn, audio_fn = mix_name
        if 'video' in self.modalities:
            video_feats = self.load_video(video_fn) # [T, H, W, 1]
        else:
            video_feats = None
        if 'audio' in self.modalities:
            audio_fn = audio_fn.split(':')[0]
            audio_npy = audio_fn.replace('.wav','.npy')
            if os.path.exists(audio_npy) == False:
                sample_rate, wav_data = wavfile.read(audio_fn)
                assert sample_rate == 16_000 and len(wav_data.shape) == 1
                audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32) # [T, F]
                audio_feats = stacker(audio_feats, self.stack_order_audio) # [T/stack_order_audio, F*stack_order_audio]
                np.save(audio_npy[:-4], audio_feats)
            else:
                audio_feats = np.load(audio_npy)
                
        else:
            audio_feats = None
        if audio_feats is not None and video_feats is not None:
            diff = len(audio_feats) - len(video_feats)
            if diff < 0:
                audio_feats = np.concatenate([audio_feats, np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)])
            elif diff > 0:
                audio_feats = audio_feats[:-diff]
        return video_feats, audio_feats

    def load_video(self, audio_name):
        feats = custom_utils.load_video(os.path.join(self.audio_root, audio_name))
        feats = self.transform(feats)
        feats = np.expand_dims(feats, axis=-1)
        return feats

    def select_noise(self):
        rand_indexes = np.random.randint(0, len(self.noise_wav), size=self.noise_num)
        noise_wav = []
        for x in rand_indexes:
            noise_wav.append(wavfile.read(self.noise_wav[x])[1].astype(np.float32))
        if self.noise_num == 1:
            return noise_wav[0]
        else:
            min_len = min([len(x) for x in noise_wav])
            noise_wav = [x[:min_len] for x in noise_wav]
            noise_wav = np.floor(np.stack(noise_wav).mean(axis=0))
            return noise_wav

    def add_noise(self, clean_wav):
        clean_wav = clean_wav.astype(np.float32)
        noise_wav = self.select_noise()
        if type(self.noise_snr) == int or type(self.noise_snr) == float:
            snr = self.noise_snr
        elif type(self.noise_snr) == tuple:
            snr = np.random.randint(self.noise_snr[0], self.noise_snr[1]+1)
        clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
        if len(clean_wav) > len(noise_wav):
            ratio = int(np.ceil(len(clean_wav)/len(noise_wav)))
            noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
        if len(clean_wav) < len(noise_wav):
            start = 0
            noise_wav = noise_wav[start: start + len(clean_wav)]
        noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1))
        adjusted_noise_rms = clean_rms / (10**(snr/20))
        adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
        mixed = clean_wav + adjusted_noise_wav

        #Avoid clipping noise
        max_int16 = np.iinfo(np.int16).max
        min_int16 = np.iinfo(np.int16).min
        if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
            if mixed.max(axis=0) >= abs(mixed.min(axis=0)): 
                reduction_rate = max_int16 / mixed.max(axis=0)
            else :
                reduction_rate = min_int16 / mixed.min(axis=0)
            mixed = mixed * (reduction_rate)
        mixed = mixed.astype(np.int16)
        return mixed


    def __getitem__(self, index):
        video_feats, audio_feats = self.load_feature(self.names[index])
        audio_feats, video_feats = torch.from_numpy(audio_feats.astype(np.float32)) if audio_feats is not None else None, torch.from_numpy(video_feats.astype(np.float32)) if video_feats is not None else None
        if self.normalize and 'audio' in self.modalities:
            with torch.no_grad():
                audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
        cluster_counts = self.load_units(index)
        labels = [self.llm_tokenizer(self.label_list[0][index], return_tensors="pt").input_ids[0]]
        labels = [torch.cat((labels[0], torch.tensor([2]).long()))]
        
        fid = self.names[index][1].split(':')[1] # split/video_id/segment_id
        
        add_sentence=""
        if self.use_topic_sentence_rate>0:
            if random.random()<self.use_topic_sentence_rate:
                topic_sentence=self.rel_path_to_topic_sentence[fid]
                add_sentence = " Topic of the speech: " + topic_sentence
        if self.use_keyword_rate>0:
            if random.random()<self.use_keyword_rate:
                keyword_sentence=self.rel_path_to_keyword[fid]
                add_sentence = add_sentence + " Keywords of the speech: " + keyword_sentence
                
        if self.use_voxmm_topic_sentence_rate>0:
            if random.random()<self.use_voxmm_topic_sentence_rate:
                video_id=fid.split('/')[-2]
                if video_id in self.video_id_to_topic_sentence:
                    voxmm_topic_sentence=self.video_id_to_topic_sentence[video_id]
                    add_sentence = add_sentence + " " + voxmm_topic_sentence
                else:
                    logger.warning(f"video id {video_id} not found in voxmm topic sentence mapping.")
                    
        if self.use_qwen_topic_sentence_rate>0:
            if random.random()<self.use_qwen_topic_sentence_rate:
                split,vid,seg=fid.split("/") 
                video_id = vid+"/"+seg # video/segment
                assert video_id in self.qwen_topic_sentence_dict,"video not found:"+video_id
                qwen_topic_sentence=self.qwen_topic_sentence_dict[video_id]
                add_sentence = add_sentence + " " +qwen_topic_sentence
            
        if self.use_multivsr_topic_sentence_rate>0:
            if random.random()<self.use_multivsr_topic_sentence_rate:
                split,vid,seg=fid.split("/") 
                assert vid in self.multivsr_topic_sentence_dict,"video not found:"+vid
                kwds = self.multivsr_topic_sentence_dict[vid]
                ks= kwds.split(", ")

                # random order
                random.shuffle(ks)
                kwds = ", ".join(ks)

                add_sentence = add_sentence + " TAGS: " + kwds + "."
        # fixed English prompt
        txt_feats = self.llm_tokenizer(
            "Recognize this English speech." + add_sentence + " Input: ",
            return_tensors="pt"
        ).input_ids[0]
        
        if self.use_uncropped_video>0:
            fid_nosplit = fid.replace("train/","").replace("test/","").replace("val/","")
            uncropped_video_path = os.path.join(self.uncropped_video_root, fid_nosplit + ".mp4")
            assert os.path.exists(uncropped_video_path)

            transcript = self.label_list[0][index]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": uncropped_video_path,
                            "min_pixels":16*32*32, 
                            "total_pixels":16*32*32,
                            "max_frames":4, 
                            "sample_fps":2,
                            "min_frames":4},
                        {"type": "text", "text": "Recognize this English speech"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": transcript}
                    ]
                }
            ]

            text = self.qwen_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            image_inputs, video_inputs, video_kwargs = process_vision_info(
                [messages],
                return_video_kwargs=True,
                image_patch_size=16,
                return_video_metadata=True
            )

            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs = list(video_inputs)
                video_metadatas = list(video_metadatas)
            else:
                video_metadatas = None

            qwen_inputs = self.qwen_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadatas,
                **video_kwargs,
                do_resize=False,
                return_tensors="pt"
            )

            input_ids = qwen_inputs["input_ids"]
            # print("INPUT IDS",input_ids.shape)
            attention_mask = qwen_inputs["attention_mask"]

            pixel_values_videos = qwen_inputs["pixel_values_videos"]
            video_grid_thw = qwen_inputs["video_grid_thw"]

            labels_qwen = input_ids.clone()

            # mask prompt tokens
            # === Fix 1: Mask prompt tokens properly for batch_size=1 ===
            labels_qwen = torch.full_like(input_ids, -100)
            # print("INPUT IDS",input_ids.shape) # 1,T

            IM_START = 77091
            IM_END = 151645

            tokens = input_ids[0].tolist()

            pos = 0
            L = len(tokens)

            while pos < L:
                if tokens[pos] == IM_START:
                    ans_start = pos + 2  # skip "<|im_start|> assistant"
                    ans_end = ans_start

                    while ans_end < L and tokens[ans_end] != IM_END:
                        ans_end += 1

                    if ans_end < L:
                        labels_qwen[0,ans_start:ans_end+2] = input_ids[0,ans_start:ans_end+2]
                        pos = ans_end

                pos += 1

            # # Extract grids if available
            # video_grid_thw = qwen_inputs.get("video_grid_thw", None)
            # attention_mask = qwen_inputs.get("attention_mask", None)

            # Compute position_ids using official function
            # position_ids, _ = get_rope_index_2(
            #     spatial_merge_size=getattr(self.qwen_processor.image_processor, "merge_size", 1),
            #     input_ids=qwen_inputs["input_ids"],
            #     video_grid_thw=video_grid_thw,
            #     attention_mask=attention_mask
            # )

            # Assign back
            # qwen_inputs["position_ids"] = position_ids
            return {
                "id": index,
                "fid": fid,

                # AVHubert inputs
                "video_source": video_feats,
                "audio_source": audio_feats,
                "cluster_counts": cluster_counts,

                # Qwen inputs
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
                "labels_qwen": labels_qwen,
                # "position_ids": position_ids,
                "label_list":labels, # wont be used in qwen case
                'text_source':[txt_feats] # wont be used in qwen case
            }
        
        
        return {"id": index, 'fid': fid, "video_source": video_feats, 'audio_source': audio_feats, "cluster_counts": cluster_counts, "label_list": labels, 'text_source':[txt_feats]}

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav, target_size, start=None):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0
        # longer utterances
        if start is None:
            start, end = 0, target_size
            if self.random_crop:
                start = np.random.randint(0, diff + 1)
                end = size - diff + start
        else:
            end = start + target_size
        return wav[start:end], start

    def collater(self, samples):
        samples = [s for s in samples if s["id"] is not None]
        if len(samples) == 0:
            return {}
    
        ############# cluster_counts ############
        cluster_counts_source = [s["cluster_counts"] for s in samples]

        ############# av_hubert ############
        audio_source, video_source = [s["audio_source"] for s in samples], [s["video_source"] for s in samples]
        if audio_source[0] is None:
            audio_source = None
        if video_source[0] is None:
            video_source = None
        if audio_source is not None:
            audio_sizes = [len(s) for s in audio_source]
        else:
            audio_sizes = [len(s) for s in video_source]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        if audio_source is not None:
            collated_audios, padding_mask, audio_starts = self.collater_audio(audio_source, audio_size)
        else:
            collated_audios, audio_starts = None, None
        if video_source is not None:
            collated_videos, padding_mask, audio_starts = self.collater_audio(video_source, audio_size, audio_starts)
        else:
            collated_videos = None

        targets_by_label = [
            [s["label_list"][i] for s in samples]
            for i in range(self.num_labels)
        ]
        
        text_instructions = [
            [s["text_source"][i] for s in samples]
            for i in range(self.num_labels)
        ]
        

        collated_texts,_ ,_  = self.collater_label(
           text_instructions, audio_size, audio_starts
        )
        
        targets_list, lengths_list, ntokens_list = self.collater_label(
            targets_by_label, audio_size, audio_starts
        )
        text_attn_mask = collated_texts[0][0] != 0
        target_attn_mask = targets_list[0][0] != 0

        source = {"audio": collated_audios, "video": collated_videos, "cluster_counts": cluster_counts_source, "text": collated_texts[0][0]}
        # ---------------- Qwen-specific inputs ----------------
        if "input_ids" in samples[0]:  # detect Qwen branch
            #source["input_ids"] = torch.stack([s["input_ids"] for s in samples], dim=0)
            #source["attention_mask"] = torch.stack([s["attention_mask"] for s in samples], dim=0)
            #source["labels_qwen"] = torch.stack([s["labels_qwen"] for s in samples], dim=0)
            source["pixel_values_videos"] = torch.stack([s["pixel_values_videos"] for s in samples], dim=0)
            source["video_grid_thw"] = torch.stack([s["video_grid_thw"] for s in samples], dim=0)

        net_input = {"source": source, "padding_mask": padding_mask, 'text_attn_mask': text_attn_mask}

        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
            "utt_id": [s['fid'] for s in samples]
        }

        if self.single_target:
            # raise NotImplementedError("single target collating not yet implemented for seq2seq case")
            batch["target_lengths"] = lengths_list[0]
            batch["ntokens"] = ntokens_list[0]
            if self.is_s2s:
                batch['target'], net_input['prev_output_tokens'] = targets_list[0][0], targets_list[0][1]
                batch['target_attn_mask'] = target_attn_mask
            else:
                batch["target"] = targets_list[0]
        else:
            batch["target_lengths_list"] = lengths_list
            batch["ntokens_list"] = ntokens_list
            batch["target_list"] = targets_list
        
        return batch
    
        
        
    def collater_audio(self, audios, audio_size, audio_starts=None):
        audio_feat_shape = list(audios[0].shape[1:])
        collated_audios = audios[0].new_zeros([len(audios), audio_size]+audio_feat_shape)
        padding_mask = (
            torch.BoolTensor(len(audios), audio_size).fill_(False) # 
        )
        start_known = audio_starts is not None
        audio_starts = [0 for _ in audios] if not start_known else audio_starts
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat(
                    [audio, audio.new_full([-diff]+audio_feat_shape, 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size, audio_starts[i] if start_known else None
                )
        if len(audios[0].shape) == 2:
            collated_audios = collated_audios.transpose(1, 2) # [B, T, F] -> [B, F, T]
        else:
            collated_audios = collated_audios.permute((0, 4, 1, 2, 3)).contiguous() # [B, T, H, W, C] -> [B, C, T, H, W]
        return collated_audios, padding_mask, audio_starts

    def collater_frm_label(
        self, targets, audio_size, audio_starts, label_rate, pad
    ):
        assert label_rate > 0
        s2f = label_rate / self.sample_rate # num label per sample
        frm_starts = [int(round(s * s2f)) for s in audio_starts]
        frm_size = int(round(audio_size * s2f))
        if not self.pad_audio:
            rem_size = [len(t) - s for t, s in zip(targets, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        targets = [t[s: s + frm_size] for t, s in zip(targets, frm_starts)]
        logger.debug(f"audio_starts={audio_starts}")
        logger.debug(f"frame_starts={frm_starts}")
        logger.debug(f"frame_size={frm_size}")

        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens


    def collater_seq_label_llm(self, targets):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        pad, eos = 0, self.llm_tokenizer.eos_token_id
        targets_ = data_utils.collate_tokens(targets, pad_idx=pad, eos_idx=eos, left_pad=False)
       
        new_targets = []
        for tar in targets:
            new_targets.append(tar[1:])

        prev_output_tokens = data_utils.collate_tokens(new_targets, pad_idx=pad, eos_idx=eos, left_pad=False, move_eos_to_beginning=False)
        
        
        prev_list = []
        for prev_tokens in prev_output_tokens:
            padding_start_idx = torch.sum(prev_tokens == 0) * -1
            if padding_start_idx == 0:
                prev_list.append(torch.cat((prev_tokens, torch.tensor([2]).long())))
            else:
                prev_tokens[padding_start_idx] = 2
                prev_list.append(torch.cat((prev_tokens, torch.tensor([0]).long())))
        
        prev_output_tokens = torch.stack(prev_list, dim=0)
        return (targets_, prev_output_tokens), lengths, ntokens


    def collater_label(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates)
        for targets, label_rate in itr:
            if label_rate == -1:  
                targets, lengths, ntokens = self.collater_seq_label_llm(targets)
            else:
                raise NotImplementedError("not yet")
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list


    def collate_tokens(self,
        values,
        pad_idx,
        eos_idxs,
        left_pad=False,
        move_eos_to_beginning=False,
        pad_to_length=None,
        pad_to_multiple=1,
        pad_to_bsz=None,
    ):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)
        if pad_to_multiple != 1 and size % pad_to_multiple != 0:
            size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

        batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
        res = values[0].new(batch_size, size).fill_(pad_idx)

        def copy_tensor(src, dst, eos_idx):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                if eos_idx is None:
                    # if no eos_idx is specified, then use the last token in src
                    dst[0] = src[-1]
                else:
                    dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)], eos_idxs[i])
        return res



    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]
