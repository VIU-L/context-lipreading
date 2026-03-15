import os
import glob
import shutil
import random
random.seed(42)
import re
from tqdm import tqdm

def read_txt_label(txt_file):
    """Extract pure text from a segment txt file."""
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("Text:"):
                text = line.split("Text:")[1].strip()
                text=text.upper()
                text = re.sub(r'(?<!\w)[^\w\s]|[^\w\s](?!\w)', '', text)
                return text
    return ""

def collect_all_youtube_ids(multivsr_root):
    """List all folders in multivsr_root = all YouTube IDs"""
    return [d for d in os.listdir(multivsr_root) if os.path.isdir(os.path.join(multivsr_root, d))]

def organize_split(multivsr_root, output_root, split_name, youtube_ids, clip_selection_func):
    """
    Copy selected clips according to clip_selection_func, and collect file.list & label.list
    clip_selection_func(list_of_clip_paths) -> list_of_clip_paths_to_keep
    """
    file_list = []
    label_list = []
    for yt_id in tqdm(youtube_ids, desc=f"Processing {split_name}"):
        src_dir = os.path.join(multivsr_root, yt_id)
        dst_dir = os.path.join(output_root, split_name, yt_id)
        os.makedirs(dst_dir, exist_ok=True)

        clips = sorted(glob.glob(os.path.join(src_dir, "*.mp4")))
        selected_clips = clip_selection_func(clips)

        for clip_path in selected_clips:
            dst_path = os.path.join(dst_dir, os.path.basename(clip_path))
            shutil.copyfile(clip_path, dst_path)

            rel_path = os.path.relpath(dst_path, output_root).replace("\\", "/")
            file_list.append(rel_path.replace(".mp4", ""))

            txt_path = clip_path.replace(".mp4", ".txt")
            label_list.append(read_txt_label(txt_path))

    return file_list, label_list

def clip_selector_train(clips):
    # Train: keep all clips
    return clips

def clip_selector_test(clips):
    # Test: only short clips
    return [c for c in clips if c.endswith("(short).mp4")]

def clip_selector_val(clips):
    # Validation: try to keep ~50% short, 50% long
    short_clips = [c for c in clips if c.endswith("(short).mp4")]
    long_clips = [c for c in clips if c.endswith("(long).mp4")]

    n = min(len(short_clips), len(long_clips))
    selected_short = random.sample(short_clips, n) if n>0 else short_clips
    selected_long = random.sample(long_clips, n) if n>0 else long_clips

    return selected_short + selected_long

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--multivsr_root", type=str, default="/mnt/sdb/yuran/av_hubert/datasets/multivsr/multivsr")
    parser.add_argument("--output_root", type=str, default="/mnt/sdb/yuran/av_hubert/datasets/multivsr/lrs3_format")
    args = parser.parse_args()

    # --- Step 1: all YouTube IDs ---
    all_youtube_ids = collect_all_youtube_ids(args.multivsr_root)
    random.shuffle(all_youtube_ids)

    num_total = len(all_youtube_ids)
    num_test = num_val = max(1, int(0.05 * num_total))
    num_train = num_total - num_val - num_test

    train_ids = all_youtube_ids[:num_train]
    val_ids   = all_youtube_ids[num_train:num_train+num_val]
    test_ids  = all_youtube_ids[num_train+num_val:]

    print(f"Total YouTube IDs: {num_total}")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    splits = {
        "train": (train_ids, clip_selector_train),
        "val":   (val_ids, clip_selector_val),
        "test":  (test_ids, clip_selector_test)
    }

    all_file_list = []
    all_label_list = []

    # --- Step 2: process each split ---
    for split_name, (yt_ids, selector) in splits.items():
        file_list, label_list = organize_split(args.multivsr_root, args.output_root, split_name, yt_ids, selector)
        all_file_list.extend(file_list)
        all_label_list.extend(label_list)
        print(f"{split_name}: {len(file_list)} clips")

    # --- Step 3: save file.list and label.list ---
    os.makedirs(args.output_root, exist_ok=True)
    with open(os.path.join(args.output_root, "file.list"), "w", encoding="utf-8") as f:
        f.write("\n".join(all_file_list) + "\n")
    with open(os.path.join(args.output_root, "label.list"), "w", encoding="utf-8") as f:
        f.write("\n".join(all_label_list) + "\n")

    print("Done. Total clips per split saved above.")