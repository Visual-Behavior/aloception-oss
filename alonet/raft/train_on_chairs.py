import argparse

from alonet.raft import Chairs2RAFT
from alonet.raft import LitRAFT
import alonet


def get_args_parser():
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser = alonet.common.add_argparse_args(parser, add_pl_args=True)
    parser = Chairs2RAFT.add_argparse_args(parser)
    parser.add_argument(
        "--use_sample", action="store_true", help="Download a sample for train process (Default: %(default)s)"
    )
    parser = LitRAFT.add_argparse_args(parser)
    return parser


def main():
    # Build parser and parse command line arguments
    args = get_args_parser().parse_args()

    # init model and dataset
    raft = LitRAFT(args)
    multi = Chairs2RAFT(args)

    raft.run_train(data_loader=multi, args=args, project="raft", expe_name="reproduce-raft-chairs")


if __name__ == "__main__":
    main()
