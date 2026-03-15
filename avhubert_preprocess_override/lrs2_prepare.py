# Copyright (c) Meta Platforms, Inc.
# All rights reserved.

import os
import subprocess
from tqdm import tqdm
import argparse

SPLITS = ["pretrain", "train", "val", "test"]

def extract_all_audio(root, ffmpeg, rank=0, nshard=1):
    """
    Extract wav audio for all splits (already trimmed videos).
    Handles LRS2 folder structure:
        pretrain -> mvlrs_v1/pretrain/
        others   -> mvlrs_v1/main/
    """
    all_entries = []

    for split in SPLITS:
        split_file = os.path.join(root, f"{split}.txt")
        assert os.path.isfile(split_file), f"Missing {split_file}"

        with open(split_file) as f:
            lines = f.readlines()

        for ln in lines:
            vid = ln.strip().split()[0]  # drop NF / MV
            all_entries.append((split, vid))

    # sharding
    num_per_shard = (len(all_entries) + nshard - 1) // nshard
    all_entries = all_entries[rank * num_per_shard : (rank + 1) * num_per_shard]

    print(f"Extracting audio for {len(all_entries)} samples")

    for split, vid in tqdm(all_entries):
        # choose correct folder
        folder = "pretrain" if split == "pretrain" else "main"
        mp4_path = os.path.join(root, folder, vid + ".mp4")
        wav_path = os.path.join(root, "audio", split, vid + ".wav")

        if not os.path.isfile(mp4_path):
            print(f"[WARN] missing video: {mp4_path}")
            continue

        os.makedirs(os.path.dirname(wav_path), exist_ok=True)

        cmd = f"{ffmpeg} -i {mp4_path} -f wav -vn -y {wav_path} -loglevel quiet"
        subprocess.call(cmd, shell=True)


def build_file_label(root):
    """
    Build file.list and label.list compatible with LRS3 / AV-HuBERT.
    Handles LRS2 folder structure as above.
    """
    file_ids = []
    labels = []

    for split in SPLITS:
        split_file = os.path.join(root, f"{split}.txt")
        with open(split_file) as f:
            lines = f.readlines()

        print(f"[{split}] collecting labels ({len(lines)})")

        for ln in tqdm(lines):
            vid = ln.strip().split()[0]  # remove NF / MV
            folder = "pretrain" if split == "pretrain" else "main"
            txt_path = os.path.join(root, folder, vid + ".txt")

            if not os.path.isfile(txt_path):
                print(f"[WARN] missing label: {txt_path}")
                continue

            with open(txt_path) as lf:
                label = lf.readlines()[0]
            label = label.split(":", 1)[-1].strip()

            file_ids.append(os.path.join(split, vid))
            labels.append(label)

    assert len(file_ids) == len(labels)

    with open(os.path.join(root, "file.list"), "w") as f:
        f.write("\n".join(file_ids) + "\n")

    with open(os.path.join(root, "label.list"), "w") as f:
        f.write("\n".join(labels) + "\n")

    print(f"Saved {len(file_ids)} samples")


def main():
    parser = argparse.ArgumentParser(
        description="LRS2 preprocessing (all splits, all steps)"
    )
    parser.add_argument("--root", type=str, default="/mnt/sdb/yuran/av_hubert/datasets/lrs2/raw/mvlrs_v1",
                        help="LRS2 root directory")
    parser.add_argument("--ffmpeg", type=str, default="/usr/bin/ffmpeg",
                        help="Path to ffmpeg binary")
    parser.add_argument("--rank", type=int, default=0,
                        help="Shard rank")
    parser.add_argument("--nshard", type=int, default=4,
                        help="Number of shards")

    args = parser.parse_args()

    print("===> Step 1: Extracting audio (all splits)")
    extract_all_audio(args.root, args.ffmpeg, args.rank, args.nshard)

    # only rank 0 writes manifests
    if args.rank == 0:
        print("===> Step 2: Building file.list and label.list")
        build_file_label(args.root)


if __name__ == "__main__":
    main()
