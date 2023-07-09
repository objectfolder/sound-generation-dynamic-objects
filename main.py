import argparse
import yaml
from easydict import EasyDict as edict
from Engine import Engine

def parse_args():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="RegNet")
    parser.add_argument("--config_location", type=str, default="./configs/default.yml")
    parser.add_argument('--eval', action='store_true', default=False, help='if True, only perform testing')
    # Data Locations
    parser.add_argument("--data_location", type=str, default='../DATA')
    parser.add_argument("--split_location", type=str, default='../DATA/split.json')
    parser.add_argument("--rgb_feature_location", type=str, default='../DATA/features/feature_rgb_bninception_dim1024_21.5fps')
    parser.add_argument("--flow_feature_location", type=str, default='../DATA/features/feature_flow_bninception_dim1024_21.5fps')
    parser.add_argument("--mel_location", type=str, default='../DATA/features/melspec_10s_22050hz')
    parser.add_argument("--audio_location", type=str, default='../DATA/features/audio_10s_22050hz')
    # Train & Evaluation
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument("--patience", type=int, default=200)
    # Exp
    parser.add_argument("--exp", type=str, default='test', help = 'The directory to save checkpoints and results')
    
    args = parser.parse_args()
    return args

def get_config(args):
    cfg_path = args.config_location
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    return edict(config)

def main():
    args = parse_args()
    cfg = get_config(args)
    engine = Engine(args, cfg)
    engine()
    
if __name__ == "__main__":
    main()