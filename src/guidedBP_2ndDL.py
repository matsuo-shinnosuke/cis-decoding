import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
from pathlib import Path

from arguments import parse_option
from model import CNN

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

    def visualize(self, input_image, target_class):
        model_output = self.model(input_image)
        self.model.zero_grad()
        pred_class = model_output.argmax().item()
        
        grad_target_map = torch.zeros(model_output.shape,
                                      dtype=torch.float)
        if target_class is not None:
            grad_target_map[0][target_class] = 1
        else:
            grad_target_map[0][pred_class] = 1
        
        model_output.backward(grad_target_map)
        
        result = self.image_reconstruction.data#[0].permute(1,2,0)
        return result.numpy()

if __name__ == "__main__":
    args = parse_option()

    X = np.load(f'{args.output_dir}/{args.fasta_file}/2nd-data.npy')
    Y = np.load(f'{args.output_dir}/{args.fasta_file}/2nd-label.npy')
    gene_name = np.load(f'{args.output_dir}/{args.fasta_file}/gene_name.npy')
    # # only test data
    # _, X, _, Y, _, gene_name = train_test_split(
    #     X, Y, gene_name, test_size=0.33, random_state=args.seed)
    X, gene_name = X[Y==1], gene_name[Y==1] # expression data
    
    ##### Inference ######
    print('Guided backpropagation ...')
    model = CNN(data_length=X.shape[1], n_channel=X.shape[2])
    model.load_state_dict(torch.load(f'{args.output_dir}/{args.fasta_file}/model.pkl'))
    model.eval()
    guided_bp = Guided_backprop(model)
    
    result_list = []
    for idx in tqdm(range(len(X)), leave=False):
        data = torch.tensor(X[idx]).float()
        data = data.unsqueeze(0).requires_grad_()

        result = guided_bp.visualize(input_image=data, target_class=1)
        result_list.append(result[0])

    result = np.array(result_list)
    np.save(f'{args.output_dir}/{args.fasta_file}/guidedBP_2ndDL_result', result)

    ###### save ######
    print('Saving results ...')
    TF_dict = open(args.data_dir+'TF_dict.txt', 'r').read().split('\n')
    
    p = Path(f'{args.output_dir}/{args.fasta_file}/2nd-weight-csv/')
    p.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(gene_name)), leave=False):
        data=result[i]
        data[data<0]=0
        data -= data.min(axis=-1, keepdims=True)
        data /= (data.max(axis=-1, keepdims=True)+1e-10)
        data = np.round(data, decimals=6)
        df = pd.DataFrame(data=data, index=TF_dict)
        df.to_csv(f'{args.output_dir}/{args.fasta_file}/2nd-weight-csv/{gene_name[i]}.csv')

