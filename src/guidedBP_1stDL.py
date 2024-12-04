import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
from pathlib import Path

from arguments import parse_option
from main_1stDL import get_FC_3layer, load_data, split_bin, detect_peak

class Guided_backprop():
    def __init__(self, model):
        self.model = model
        self.image_reconstruction = None # store R0
        self.activation_maps = []  # store f1, f2, ... 
        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def first_layer_hook_fn(module, grad_in, grad_out):
            self.image_reconstruction = grad_in[0] 

        def forward_hook_fn(module, input, output):
            self.activation_maps.append(output)

        def backward_hook_fn(module, grad_in, grad_out):
            grad = self.activation_maps.pop() 
            # for the forward pass, after the ReLU operation, 
            # if the output value is positive, we set the value to 1,
            # and if the output value is negative, we set it to 0.
            grad[grad > 0] = 1 
            
            # grad_out[0] stores the gradients for each feature map,
            # and we only retain the positive gradients
            positive_grad_out = torch.clamp(grad_out[0], min=0.0)
            new_grad_in = positive_grad_out * grad

            return (new_grad_in,)

        modules = list(self.model.features.named_children())

        # travese the modules，register forward hook & backward hook
        # for the ReLU
        for name, module in modules:
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook_fn)
                module.register_full_backward_hook(backward_hook_fn)

        # register backward hook for the first conv layer
        first_layer = modules[0][1] 
        first_layer.register_full_backward_hook(first_layer_hook_fn)

    def visualize(self, input_image, target_position):
        model_output = self.model(input_image)
        self.model.zero_grad()
        
        grad_target_map = torch.zeros(model_output.shape, dtype=torch.float).to(args.device)
        grad_target_map[0, target_position, 1] = 1
        
        model_output.backward(grad_target_map)
        
        result = self.image_reconstruction.data#[0].permute(1,2,0)
        return result.cpu().numpy()

if __name__ == "__main__":
    args = parse_option()

    ######## Loading data #########
    gene_name, data, data_origin = load_data(data_dir=args.data_dir, 
                           fasta_file_name=args.fasta_file, 
                           gene_length=args.length,
                           output_dir=args.output_dir)
    # TF_dict = open(args.data_dir+'TF_dict.txt', 'r').read().split('\n')

    ######## Splitting bin #########
    gene_data_bin, gene_data_origin_bin = split_bin(X=data, 
                                                    X_origin=data_origin,
                                                    output_dir=args.output_dir,
                                                    fasta_file_name=args.fasta_file,
                                                    bin=args.bin, 
                                                    walk=args.walk)

    ##### Inference ######
    print('Guided backpropagation ...')
    TF_name = args.TF_name
    print(TF_name)

    model = get_FC_3layer(bin=args.bin).to(args.device)
    model.load_state_dict(torch.load(f'{args.model_dir}/{TF_name}.pkl', map_location=torch.device(args.device)))
    guided_bp = Guided_backprop(model)
    pred = np.load(f'{args.output_dir}/{args.fasta_file}/1st-pred/{TF_name}.npy')

    result_gene_list = []
    for idx in tqdm(range(len(gene_data_bin)), leave=False):
        data = torch.tensor(gene_data_bin[idx]).float().to(args.device)
        data = data.unsqueeze(0).requires_grad_()
        target = np.where(pred[idx]>=args.threshold)[0]

        result = guided_bp.visualize(input_image=data, target_position=target)
        result_gene_list.append(result[0])

    result = np.array(result_gene_list)

    ###### save ######
    print('Saving results ...')    
    Path(f'{args.output_dir}/{args.fasta_file}/1st-weight-csv/{TF_name}/').mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(len(gene_name)), leave=False):
        data = result[i].reshape(-1, args.bin)

        data[data<0]=0
        data -= data.min(axis=-1, keepdims=True)
        data /= (data.max(axis=-1, keepdims=True)+1e-10)
        data = np.round(data, decimals=2)

        index = [elem for item in gene_data_origin_bin[i] for elem in (item+'_A', ' '*len(item)+'_T', ' '*len(item)+'_G', ' '*len(item)+'_C')]

        df = pd.DataFrame(data=data, index=index)
        df.to_csv(f'{args.output_dir}/{args.fasta_file}/1st-weight-csv/{TF_name}/{gene_name[i]}.csv')