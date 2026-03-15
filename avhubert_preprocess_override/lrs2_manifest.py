# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from gen_subword import gen_vocab

def main():
    import argparse
    parser = argparse.ArgumentParser(description='LRS2/LRS3 tsv preparation', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lrs2', type=str, help='LRS2 root dir', 
                        default="/mnt/sdb/yuran/av_hubert/datasets/lrs2/raw/mvlrs_v1")
    parser.add_argument('--vocab-size', type=int, default=1000, help='SentencePiece vocab size')
    args = parser.parse_args()

    # Files
    file_list = f"{args.lrs2}/file.list"
    label_list = f"{args.lrs2}/label.list"
    nframes_audio_file = f"{args.lrs2}/nframes.audio"
    nframes_video_file = f"{args.lrs2}/nframes.video"

    # Check required files exist
    for f in [file_list, label_list, nframes_audio_file, nframes_video_file]:
        assert os.path.isfile(f), f"{f} not exist -> run dataset_prepare.py / count_frames.py first"

    # Generate sentencepiece vocab
    print(f"Generating sentencepiece units...")
    vocab_dir = Path(args.lrs2) / f"spm{args.vocab_size}"
    vocab_dir.mkdir(exist_ok=True)
    spm_prefix = f"spm_unigram{args.vocab_size}"

    with NamedTemporaryFile(mode="w") as tmpf:
        for ln in open(label_list).readlines():
            tmpf.write(ln.strip().lower() + "\n")
        gen_vocab(Path(tmpf.name), vocab_dir / spm_prefix, 'unigram', args.vocab_size)

    vocab_path = (vocab_dir / spm_prefix).as_posix() + '.txt'

    # Audio/video directories
    audio_dir = f"{args.lrs2}/audio"
    video_dir = f"{args.lrs2}/video"

    # Helper to write tsv/wrd files
    def setup_target(target_dir, train, valid, test):
        for name, data in zip(['train', 'valid', 'test'], [train, valid, test]):
            with open(f"{target_dir}/{name}.tsv", 'w') as fo:
                fo.write('/\n')
                for fid, _, nf_audio, nf_video in data:
                    fo.write('\t'.join([
                        fid,
                        os.path.abspath(f"{video_dir}/{fid}.mp4"),
                        os.path.abspath(f"{audio_dir}/{fid}.wav"),
                        str(nf_video),
                        str(nf_audio)
                    ]) + '\n')
            with open(f"{target_dir}/{name}.wrd", 'w') as fo:
                for _, label, _, _ in data:
                    fo.write(f"{label}\n")
        shutil.copyfile(vocab_path, f"{target_dir}/dict.wrd.txt")

    # Read all dataset info
    fids = [x.strip() for x in open(file_list).readlines()]
    labels = [x.strip().lower() for x in open(label_list).readlines()]
    nfs_audio = [x.strip() for x in open(nframes_audio_file).readlines()]
    nfs_video = [x.strip() for x in open(nframes_video_file).readlines()]

    # Split datasets according to your LRS2 structure
    train_big, train_small, valid, test = [], [], [], []

    for fid, label, nf_audio, nf_video in zip(fids, labels, nfs_audio, nfs_video):
        part = fid.split('/')[0]  # pretrain / train / val / test

        if part == 'test':
            test.append([fid, label, nf_audio, nf_video])
        elif part == 'val':
            valid.append([fid, label, nf_audio, nf_video])
        elif part == 'train':
            train_small.append([fid, label, nf_audio, nf_video])
            train_big.append([fid, label, nf_audio, nf_video])
        elif part == 'pretrain':
            train_big.append([fid, label, nf_audio, nf_video])

    # Create small dataset (train only from train)
    dir_small = f"{args.lrs2}/small_data"
    print(f"Setting up small dataset in {dir_small}")
    os.makedirs(dir_small, exist_ok=True)
    setup_target(dir_small, train_small, valid, test)

    # Create big dataset (train from pretrain + train)
    dir_big = f"{args.lrs2}/big_data"
    print(f"Setting up big dataset in {dir_big}")
    os.makedirs(dir_big, exist_ok=True)
    setup_target(dir_big, train_big, valid, test)

if __name__ == '__main__':
    main()
