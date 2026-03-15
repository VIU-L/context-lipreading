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
    parser = argparse.ArgumentParser(description='Video-only LRS2/LRS3 tsv preparation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, help='Dataset root dir',
                        default="/mnt/sdb/yuran/av_hubert/datasets/multivsr/lrs3_format")
    parser.add_argument('--vocab-size', type=int, default=1000, help='SentencePiece vocab size')
    args = parser.parse_args()

    # Files
    file_list = f"{args.dataset}/file.list"
    label_list = f"{args.dataset}/label.list"
    nframes_video_file = f"{args.dataset}/nframes.video"

    # Check required files exist
    for f in [file_list, label_list, nframes_video_file]:
        assert os.path.isfile(f), f"{f} not exist -> run dataset_prepare.py / count_frames.py first"

    # Create dummy audio file
    audio_dir = f"{args.dataset}/audio"
    os.makedirs(audio_dir, exist_ok=True)
    dummy_audio_path = os.path.abspath(f"{audio_dir}/dummy.wav")
    if not os.path.isfile(dummy_audio_path):
        with open(dummy_audio_path, 'wb') as f:
            f.write(b'')  # empty dummy audio file

    # Generate sentencepiece vocab
    print(f"Generating sentencepiece units...")
    vocab_dir = Path(args.dataset) / f"spm{args.vocab_size}"
    vocab_dir.mkdir(exist_ok=True)
    spm_prefix = f"spm_unigram{args.vocab_size}"

    with NamedTemporaryFile(mode="w") as tmpf:
        for ln in open(label_list).readlines():
            tmpf.write(ln.strip().lower() + "\n")
        gen_vocab(Path(tmpf.name), vocab_dir / spm_prefix, 'unigram', args.vocab_size)

    vocab_path = (vocab_dir / spm_prefix).as_posix() + '.txt'

    # Video directory
    video_dir = f"{args.dataset}/video"

    # Helper to write tsv/wrd files
    def setup_target(target_dir, train, valid, test):
        for name, data in zip(['train', 'valid', 'test'], [train, valid, test]):
            with open(f"{target_dir}/{name}.tsv", 'w') as fo:
                fo.write('/\n')
                for fid, label, nf_video in data:
                    fo.write('\t'.join([
                        fid,
                        os.path.abspath(f"{video_dir}/{fid}.mp4"),
                        dummy_audio_path,   # dummy audio path
                        str(nf_video),
                        str(int(int(nf_video)/640)) # dummy audio frame count
                    ]) + '\n')
            with open(f"{target_dir}/{name}.wrd", 'w') as fo:
                for _, label, _ in data:
                    fo.write(f"{label}\n")
        shutil.copyfile(vocab_path, f"{target_dir}/dict.wrd.txt")

    # Read all dataset info
    fids = [x.strip() for x in open(file_list).readlines()]
    labels = [x.strip().lower() for x in open(label_list).readlines()]
    nfs_video = [x.strip() for x in open(nframes_video_file).readlines()]

    # Split datasets according to your dataset structure
    train, valid, test = [], [], []

    for fid, label, nf_video in zip(fids, labels, nfs_video):
        part = fid.split('/')[0]  # train / val / test

        if part == 'test':
            test.append([fid, label, nf_video])
        elif part == 'val':
            valid.append([fid, label, nf_video])
        elif part == 'train':
            train.append([fid, label, nf_video])

    # Create dataset folder
    target_dir = f"{args.dataset}/data"
    print(f"Setting up dataset in {target_dir}")
    os.makedirs(target_dir, exist_ok=True)
    setup_target(target_dir, train, valid, test)

if __name__ == '__main__':
    main()
