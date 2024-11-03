import yaml
import sys
import argparse

def update_yaml(args, input_file, output_file):
    
    with open(input_file, 'r') as file:
        data = yaml.safe_load(file)

    data['model']['lora_rank'] = int(args.lora_rank)
    data['model']['lora_alpha'] = float(args.lora_alpha)
    data['method'] = args.method
    data['batch_size'] = int(args.batch_size)
    data['n_val'] = int(args.n_val)
    data['fracinv'] = float(args.fracinv)
    data['optimizer']['lr'] = float(args.lr)

    with open(output_file, 'w') as file:
        yaml.safe_dump(data, file)

def main():
    parser = argparse.ArgumentParser(description='Update YAML configuration file')
    parser.add_argument('--input_file', type=str, help='Input YAML file')

    parser.add_argument('--method', type=str, required=True, help='Method to use')
    parser.add_argument('--lora_rank', type=int, required=True, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, required=True, help='LoRA alpha')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--n_val', type=int, required=True, help='Number of validation samples')
    parser.add_argument('--fracinv', type=float, required=True, help='Fraction invertible')
    parser.add_argument('--lr', type=float, required=True, help='Fraction invertible')


    args = parser.parse_args()

    output_file = f"{args.input_file.rstrip('.yaml')}_Exp_{args.method}_LR{args.lr}_LoRA_R{args.lora_rank}_alpha{args.lora_alpha}_bs{args.batch_size}_nval{args.n_val}_fracinv{args.fracinv}.yaml"

    update_yaml(args, args.input_file, output_file)

if __name__ == "__main__":
    main()