

import numpy as np



def compute_TracIN_GC_per_iter(model, device, batch_data, validation_loader, optimizer):

    per_val=False
    return_tracin_and_similarity=True

    # data, target = batch_data
    print(batch_data.keys())
    # batch_size = validation_loader.batch_size
    # TODO: for Tong - find a way to get batch size 
    batch_size = batch_data['input_ids'].shape[0]
    logger.info("***** Running %s *****", "validation_loader")
    logger.info("  Num examples = %d", len(validation_loader.dataset))
    logger.info("  Batch size = %d", batch_size)
    optimizer.zero_grad()

    trainable_layers = [model.lm_head]

    print("trainable_layers: ", trainable_layers)

    dLdZ_a_val_lst = []
    for step, inputs in enumerate(validation_loader):
        print(inputs['input_ids'].shape)
        outputs = model(**inputs)
        val_loss = outputs.loss
        val_pre_acts = [layer.pre_activation for layer in trainable_layers]
        Z_grad_val = torch.autograd.grad(val_loss, val_pre_acts, retain_graph=True)
        for layer, zgrad_val in zip(trainable_layers, Z_grad_val):
            decompose_result = layer.pe_grad_gradcomp(zgrad_val, per_sample=True)
            print("decompose_result", decompose_result[0].shape, decompose_result[1].shape)
            dLdZ_a_val_lst = update_list(dLdZ_a_val_lst, decompose_result)
        
        if step == 1:
            break

    optimizer.zero_grad()
    # Compute individual training loss
    output_train = model(**batch_data)
    train_loss = output_train.loss
    print("train_loss", train_loss)
    mean_train_loss = train_loss.mean()
    pre_acts = [layer.pre_activation for layer in trainable_layers]
    Z_grad = torch.autograd.grad(mean_train_loss, pre_acts, retain_graph=False)
    dLdZ_a_train_lst = []
    for layer, zgrad in zip(trainable_layers, Z_grad):
        decompose_result = layer.pe_grad_gradcomp(zgrad, per_sample=True)
        dLdZ_a_train_lst = update_list(dLdZ_a_train_lst, decompose_result)
    # Compute TracIN score
    tracin_local_score = np.zeros( (batch_size, n_val) ) if per_val else np.zeros(batch_size)

    if return_tracin_and_similarity:
        similarity_local_score = np.zeros( (batch_size, batch_size) )

    for layer, (dLdZ, a), (dLdZ_val, a_val) in zip(trainable_layers, dLdZ_a_train_lst, dLdZ_a_val_lst):
        dLdZ = dLdZ.detach()
        a = a.detach()

        print('dLdZ.shape={}, a.shape={}, dLdZ_val.shape={}, a_val.shape={}'.format(
            dLdZ.shape, a.shape, dLdZ_val.shape, a_val.shape
        ))

        dot_prod = grad_dotprod(dLdZ, a, dLdZ_val, a_val)

        if per_val:
            tracin_local_score += (dot_prod).cpu().detach().numpy()
        else:
            tracin_local_score += ((dot_prod).mean(dim=1)).cpu().detach().numpy()

        if return_tracin_and_similarity:
            dot_prod = grad_dotprod(dLdZ, a, dLdZ, a)
            similarity_local_score += (dot_prod).cpu().detach().numpy()

    if return_tracin_and_similarity:
        return tracin_local_score, similarity_local_score
    else:
        return tracin_local_score


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


def grad_dotprod_sequential(A1, B1, A2, B2):

    (b, t, p), (_, _, d) = A1.size(), B1.size()
    nval, _, _ = A2.size()

    if 2*b*nval*t**2 < (b+nval)*p*d:

        A2, B2 = A2.transpose(-1, -2), B2.transpose(-1, -2)

        A1_expanded = A1.unsqueeze(1)
        A2_expanded = A2.unsqueeze(0)
        B1_expanded = B1.unsqueeze(1)
        B2_expanded = B2.unsqueeze(0)

        # Memory consumption: 2*b*nval*T^2
        # A_dotprod = torch.matmul(A1_expanded, A2_expanded) # Shape: [b, nval, T, T]
        # B_dotprod = torch.matmul(B1_expanded, B2_expanded) # Shape: [b, nval, T, T]
        A_dotprod = _chunked_matmul(A1_expanded, A2_expanded, chunk_size=128)
        B_dotprod = _chunked_matmul(B1_expanded, B2_expanded, chunk_size=128)

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




