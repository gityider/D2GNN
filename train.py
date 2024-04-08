
import argparse
import torch
import os
import utils

import Dataset
import functions
from Model import D2GNN
from Coach import Coach

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


def main(args):
    utils.set_seed(args.seed)

    args.data = os.path.join(
        args.data_dir_path, args.dataset, "data_" + args.dataset + ".pkl"
    )

    # load data
    log.debug("Loading data from '%s'." % args.data)

    data = utils.load_pkl(args.data)
    log.info("Loaded data.")

    trainset = Dataset.Dataset(data["train"], args)
    devset = Dataset.Dataset(data["dev"], args)
    testset = Dataset.Dataset(data["test"], args)

    log.debug("Building model...")
    
    model_file = "./model_checkpoints/model.pt"
    model = D2GNN(args).to(args.device)
    opt = functions.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
    opt.set_parameters(model.parameters(), args.optimizer)
    sched = opt.get_scheduler(args.scheduler)

    coach = Coach(trainset, devset, testset, model, opt, sched, args)
    if not args.from_begin:
        ckpt = torch.load(model_file)
        coach.load_ckpt(ckpt)
        print("Training from checkpoint...")

    # Train
    log.info("Start training...")
    ret = coach.train()

    # Save.
    checkpoint = {
        "best_dev_f1": ret[0],
        "best_epoch": ret[1],
        "best_state": ret[2],
    }
    torch.save(checkpoint, model_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument(
        "--dataset",
        type=str,
        default="iemocap",
        help="Dataset name:iemocap",
    )
    parser.add_argument(
        "--data_dir_path", type=str, help="Dataset directory path", default="./data"
    )
    parser.add_argument(
        "--from_begin", action="store_false", help="Training from begin.", default=True
    )
    parser.add_argument("--device", type=str, default="cuda", help="Computing device.")
    parser.add_argument(
        "--epochs", default=50, type=int, help="Number of training epochs."
    )
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "rmsprop", "adam", "adamw"],
        help="Name of optimizer.",
    )
    parser.add_argument(
        "--scheduler", type=str, default="reduceLR", help="Name of scheduler."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0002, help="Learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8, help="Weight decay."
    )
    parser.add_argument(
        "--max_grad_value",
        default=-1,
        type=float,
        help="""If the norm of the gradient vector exceeds this,
                        normalize it to have the norm equal to max_grad_norm""",
    )
    parser.add_argument("--drop_rate", type=float, default=0.5, help="Dropout rate.")
    parser.add_argument(
        "--wp",
        type=int,
        default=5,
        help="Past context window size. Set wp to -1 to use all the past context.",
    )
    parser.add_argument(
        "--wf",
        type=int,
        default=5,
        help="Future context window size. Set wp to -1 to use all the future context.",
    )
    parser.add_argument("--n_speakers", type=int, default=2, help="Number of speakers.")
    parser.add_argument("--n_classes", type=int, default=6, help="Number of classes.")
    parser.add_argument("--n_views", type=int, default=3, help="Number of modalities.")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of two layer GCN.")
    parser.add_argument("--rnn",type=str, default="lstm",)
    parser.add_argument("--class_weight", action="store_true", default=False, help="Use class weights in nll loss.")
    parser.add_argument("--modalities", type=str, default="atv", help="Modalities",)
    parser.add_argument("--seqcontext_nlayer", type=int, default=2)
    parser.add_argument("--gnn_nheads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=144, help="Random seed.")

    args = parser.parse_args()

    args.dataset_embedding_dims = {
        "iemocap": {
            "a": 100,
            "t": 768,
            "v": 512,
            "at": 100 + 768,
            "tv": 768 + 512,
            "av": 612,
            "atv": 100 + 768 + 512,
        },
    }

    log.debug(args)

    main(args)
