import torch
import numpy as np
import less
import time


def find_GClayers(module):

    GC_layers = []

    for layer_str in dir(module):
        layer = getattr(module, layer_str)
        if type(layer) in [less.layers.lora_layers.GCLoRALinear, less.layers.linear.GCLinear]:
            # print('Found GC Layer: {}'.format(layer_str))
            GC_layers.append( layer )

    if hasattr(module,'children'):
        for immediate_child_module in module.children():
            GC_layers = GC_layers + find_GClayers(immediate_child_module)
            
    return GC_layers



def compute_GradProd_GC_per_iter(model, device, batch_train, validation_loader, optimizer, trainable_layers, 
                               per_val=False, return_tracin_and_similarity=True):

    # Get first batch from validation loader
    batch_val = next(iter(validation_loader))

    # Get the batch size of the validation and training batches
    val_bs = batch_val['input_ids'].shape[0]
    train_bs = batch_train['input_ids'].shape[0]

    optimizer.zero_grad()

    # Get maximum sequence length from both batches
    max_seq_len = max(
        batch_train['input_ids'].shape[1],
        batch_val['input_ids'].shape[1]
    )

    # Pad training batch if needed
    if batch_train['input_ids'].shape[1] < max_seq_len:
        pad_length = max_seq_len - batch_train['input_ids'].shape[1]
        batch_train = {
            'input_ids': torch.nn.functional.pad(batch_train['input_ids'], (0, pad_length), value=0),
            'attention_mask': torch.nn.functional.pad(batch_train['attention_mask'], (0, pad_length), value=0),
            'labels': torch.nn.functional.pad(batch_train['labels'], (0, pad_length), value=-100)  # Use -100 for labels padding
        }

    # Pad validation batch if needed
    if batch_val['input_ids'].shape[1] < max_seq_len:
        pad_length = max_seq_len - batch_val['input_ids'].shape[1]
        batch_val = {
            'input_ids': torch.nn.functional.pad(batch_val['input_ids'], (0, pad_length), value=0),
            'attention_mask': torch.nn.functional.pad(batch_val['attention_mask'], (0, pad_length), value=0),
            'labels': torch.nn.functional.pad(batch_val['labels'], (0, pad_length), value=-100)  # Use -100 for labels padding
        }

    # Separate labels from inputs
    combined_labels = torch.cat([batch_train['labels'], batch_val['labels']], dim=0)
    combined_inputs = {
        k: torch.cat([batch_train[k], batch_val[k]], dim=0) 
        for k in batch_train.keys() if k != 'labels'  # Exclude labels
    }
    
    # Free memory from individual batches
    del batch_train, batch_val
    torch.cuda.empty_cache()

    # Forward pass without loss computation
    outputs = model(**combined_inputs)
    logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
    
    # Compute per-sample losses manually
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')  # Use 'none' to get per-sample losses
    # Reshape logits and labels for loss computation
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = combined_labels[..., 1:].contiguous()
    # Compute loss for each position
    per_position_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    # Reshape to [batch_size, seq_len-1]
    per_position_loss = per_position_loss.view(shift_labels.size())
    # Mask out padding tokens (where labels == -100)
    mask = (shift_labels != -100).float()
    # Get per-sample loss by averaging over sequence length
    loss = (per_position_loss * mask).sum(dim=-1) / mask.sum(dim=-1)
    
    pre_acts = [layer.pre_activation for layer in trainable_layers]
    
    # Single backward pass using mean loss for gradient computation
    Z_grad = torch.autograd.grad(loss.mean(), pre_acts, retain_graph=True)

    dLdZ_a_train_lst = []
    dLdZ_a_val_lst = []
    
    for layer, zgrad in zip(trainable_layers, Z_grad):

        decompose_results = layer.pe_grad_gradcomp(zgrad, per_sample=True)

        # Pre-allocate lists with known size
        train_results = [None] * len(decompose_results)
        val_results = [None] * len(decompose_results)

        # Single loop with direct indexing
        for i, (dLdZ, a) in enumerate(decompose_results):
            # Use torch.split instead of slicing for better memory efficiency
            dLdZ_train, dLdZ_val = torch.split(dLdZ, [train_bs, dLdZ.size(0) - train_bs])
            a_train, a_val = torch.split(a, [train_bs, a.size(0) - train_bs])
            
            train_results[i] = (dLdZ_train, a_train)
            val_results[i] = (dLdZ_val, a_val)

        dLdZ_a_train_lst.extend(train_results)
        dLdZ_a_val_lst.extend(val_results)


    # Compute Gradient Dot-Product between training and validation batches
    grad_dotproduct_score = np.zeros((train_bs, val_bs)) if per_val else np.zeros(train_bs)

    # Compute pairwise similarity between training samples
    if return_tracin_and_similarity:
        similarity_local_score = np.zeros((train_bs, train_bs))

    assert len(dLdZ_a_train_lst) == len(dLdZ_a_val_lst)

    for (dLdZ, a), (dLdZ_val, a_val) in zip(dLdZ_a_train_lst, dLdZ_a_val_lst):

        dot_prod = grad_dotprod(dLdZ, a, dLdZ_val, a_val)

        if per_val:
            grad_dotproduct_score += ((dot_prod).float()).cpu().detach().numpy()
        else:
            grad_dotproduct_score += ((dot_prod).mean(dim=1).float()).cpu().detach().numpy()

        if return_tracin_and_similarity:
            dot_prod = grad_dotprod(dLdZ, a, dLdZ, a)
            similarity_local_score += ((dot_prod).float()).cpu().detach().numpy()

    return grad_dotproduct_score, similarity_local_score



def update_list(original, input_element):
    # Check if the input is a list
    if isinstance(input_element, list):
        # Concatenate with the original list
        return original + input_element
    else:
        # Append to the original list
        original.append(input_element)
        return original


def grad_dotprod(A1, B1, A2, B2) -> torch.Tensor:
    """Compute gradient sample norm for the weight matrix in a linear layer."""
    if A1.dim() == 2 and B1.dim() == 2:
        return grad_dotprod_non_sequential(A1, B1, A2, B2)
    elif A1.dim() == 3 and B1.dim() == 3:
        return grad_dotprod_sequential(A1, B1, A2, B2)
    else:
        raise ValueError(f"Unexpected input shape: {A1.size()}, grad_output shape: {B1.size()}")


def grad_dotprod_non_sequential(A1, B1, A2, B2):

    dot_prod_1 = torch.matmul(A1, A2.T)
    dot_prod_2 = torch.matmul(B1, B2.T)
    dot_prod = dot_prod_1*dot_prod_2

    return dot_prod


def grad_dotprod_sequential(A1, B1, A2, B2, chunk_size=1024):

    (b, t, p), (_, _, d) = A1.size(), B1.size()
    nval, _, _ = A2.size()

    # if 2*b*nval*t**2 < (b+nval)*p*d:
    if False:

        """
        This part is not used because it is slower than the case without chunking.
        """

        A2, B2 = A2.transpose(-1, -2), B2.transpose(-1, -2)

        A1_expanded = A1.unsqueeze(1)
        A2_expanded = A2.unsqueeze(0)
        B1_expanded = B1.unsqueeze(1)
        B2_expanded = B2.unsqueeze(0)

        # Memory consumption: 2*b*nval*T^2
        # A_dotprod = torch.matmul(A1_expanded, A2_expanded) # Shape: [b, nval, T, T]
        # B_dotprod = torch.matmul(B1_expanded, B2_expanded) # Shape: [b, nval, T, T]
        A_dotprod = _chunked_matmul(A1_expanded, A2_expanded, chunk_size=chunk_size)
        B_dotprod = _chunked_matmul(B1_expanded, B2_expanded, chunk_size=chunk_size)

        return (A_dotprod * B_dotprod).sum(dim=(2, 3))
    
    else:

        # [b, p, T] * [b, T, d]
        A = torch.bmm(B1.permute(0, 2, 1), A1).flatten(start_dim=1) # Shape: [b, p*d]
        B = torch.bmm(B2.permute(0, 2, 1), A2).flatten(start_dim=1) # Shape: [nval, p*d]

        return torch.matmul(A, B.T)


def _chunked_matmul(A1, A2, chunk_size=128):
    """
    Performs matrix multiplication in chunks for memory efficiency.

    Parameters:
    A1 (torch.Tensor): The first tensor with shape [n1, c1, h1, w1]
    A2 (torch.Tensor): The second tensor with shape [n2, c2, w2, h2]
    chunk_size (int): The size of each chunk to be multiplied

    Returns:
    torch.Tensor: The result of the matrix multiplication with shape [n1, c2, h1, h2]
    """
    # Validate input shapes
    if A1.shape[-1] != A2.shape[-2]:
        raise ValueError(f"Inner dimensions must match for matrix multiplication, got {A1.shape[-1]} and {A2.shape[-2]}")

    # Determine output shape
    n1, c1, h1, w1 = A1.shape
    n2, c2, w2, h2 = A2.shape

    if w1 != w2:
        raise ValueError(f"Inner matrix dimensions must agree, got {w1} and {w2}")

    # Prepare the result tensor on the same device as the inputs
    result = torch.zeros(n1, c2, h1, h2, device=A1.device, dtype=A1.dtype)

    # Perform the multiplication in chunks
    for start in range(0, w1, chunk_size):
        end = min(start + chunk_size, w1)
        A1_chunk = A1[:, :, :, start:end]  # [8, 1, 1024, chunk_size]
        A2_chunk = A2[:, :, start:end, :]  # [1, 8, chunk_size, 1024]

        # Multiply the chunks
        result += torch.matmul(A1_chunk, A2_chunk)

    return result


def greedy_selection(scores, interaction_matrix, K):
    """
    Select K data points based on the highest scores, dynamically updating scores
    by subtracting interactions with previously selected data points.

    Parameters:
    - scores: A numpy array of initial scores for each data point.
    - interaction_matrix: A numpy matrix of pairwise interactions between data points.
    - K: The number of data points to select.

    Returns:
    - selected_indices: Indices of the selected data points.
    """
    # Ensure scores is a mutable numpy array to update it in-place
    scores = scores.copy()
    selected_indices = []

    for _ in range(K):
        # Select the index with the highest score
        idx_max = np.argmax(scores)
        selected_indices.append(idx_max)

        # Update scores by subtracting interactions with the selected data point
        scores -= interaction_matrix[idx_max, :]

        # Set the score of the selected data point to a very large negative value
        # to ensure it's not selected again
        scores[idx_max] = -np.inf

    return selected_indices




