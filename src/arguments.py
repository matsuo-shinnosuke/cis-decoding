import argparse
from pathlib import Path


def parse_option():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='dataset/', type=str, help='data directory name')
    parser.add_argument('--fasta_file', default='sample_data.fa', type=str, help='fasta file name')
    parser.add_argument('--label_file', default='sample_label.txt', type=str, help='label file name')
    parser.add_argument('--length', default=2001, type=int, help='gene length')
    parser.add_argument('--model_dir', default='model/', type=str, help='model directory name')
    parser.add_argument('--output_dir', default='result/',  type=str, help="output directory name")

    parser.add_argument('--walk', default=2, type=int, help='walk bp size')
    parser.add_argument('--bin', default=31, type=int, help='bin bp size')
    parser.add_argument('--bin_peak', default=25, type=int, help='bin bp size to detect peak')
    parser.add_argument('--threshold', default=0.8, type=float, help='bin bp size')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--model', default='CNN', choices=['CNN', 'Transformer'], type=str)

    parser.add_argument('--TF_name', default='ABF2_col_v3a', type=str)

    args = parser.parse_args()

    p = Path(args.output_dir)
    p.mkdir(parents=True, exist_ok=True)

    return args
