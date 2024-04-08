import argparse

from tqdm import tqdm
import pickle
import os
import json
import pandas as pd
import numpy as np

import utils

from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer("paraphrase-distilroberta-base-v1")

log = utils.get_logger()

class IEMOCAP_Sample:
    def __init__(self, vid, speaker, label, text, audio, visual, sentence):
        self.vid = vid
        self.speaker = speaker
        self.label = label
        self.text = text
        self.audio = audio
        self.visual = visual
        self.sentence = sentence
        self.sbert_sentence_embeddings = sbert_model.encode(sentence)

def get_iemocap():
    utils.set_seed(args.seed)

    if args.dataset == "iemocap":
        (
            video_ids,
            video_speakers,
            video_labels,
            video_text,
            video_audio,
            video_visual,
            video_sentence,
            trainVids,
            test_vids,
        ) = pickle.load(
            open("./data/iemocap/IEMOCAP_features.pkl", "rb"), encoding="latin1"
        )

    train, dev, test = [], [], []
    dev_size = int(len(trainVids) * 0.1)
    train_vids, dev_vids = trainVids[dev_size:], trainVids[:dev_size]

    for vid in tqdm(train_vids, desc="train"):
        train.append(
            IEMOCAP_Sample(
                vid,
                video_speakers[vid],
                video_labels[vid],
                video_text[vid],
                video_audio[vid],
                video_visual[vid],
                video_sentence[vid],
            )
        )
    for vid in tqdm(dev_vids, desc="dev"):
        dev.append(
            IEMOCAP_Sample(
                vid,
                video_speakers[vid],
                video_labels[vid],
                video_text[vid],
                video_audio[vid],
                video_visual[vid],
                video_sentence[vid],
            )
        )
    for vid in tqdm(test_vids, desc="test"):
        test.append(
            IEMOCAP_Sample(
                vid,
                video_speakers[vid],
                video_labels[vid],
                video_text[vid],
                video_audio[vid],
                video_visual[vid],
                video_sentence[vid],
            )
        )
    log.info("train vids:")
    log.info(sorted(train_vids))
    log.info("dev vids:")
    log.info(sorted(dev_vids))
    log.info("test vids:")
    log.info(sorted(test_vids))

    return train, dev, test

def main(args):
    if args.dataset == "iemocap":
        train, dev, test = get_iemocap()
        data = {"train": train, "dev": dev, "test": test}
        utils.save_pkl(data, "./data/iemocap/data_iemocap.pkl")

    log.info("number of train samples: {}".format(len(train)))
    log.info("number of dev samples: {}".format(len(dev)))
    log.info("number of test samples: {}".format(len(test)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess.py")

    parser.add_argument(
        "--dataset",
        type=str,
        default="iemocap",
        help="Dataset name:iemocap,meld",
    )

    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Dataset directory"
    )
    parser.add_argument("--seed", type=int, default=24, help="Random seed.")
    args = parser.parse_args()

    main(args)
