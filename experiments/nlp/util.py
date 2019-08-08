import torch
import numpy as np
import warnings

def pred(s: str, tokenizer, clf, device='cpu'):
    '''Predict for a string given tokenizer and (returns class 1 - the positive class)
    '''
    with torch.no_grad():
        # print('encoded', tokenizer.convert_ids_to_tokens(tokenizer.encode(s)))
        input_ids = torch.tensor(tokenizer.encode(s), device=device).unsqueeze(0)  # Batch size 1
        pred = clf(input_ids)[0].detach().cpu().softmax(dim=1).numpy().flatten()
    return pred[1]


def predict_masked(masked_predictor, tokenizer,
                   text: str="[CLS] This great movie was very good . [SEP] I thoroughly enjoyed it [SEP]",
                   masked_index: int=None,
                   masked_str: str=None,
                   device: str='cpu'):
    '''Predict what to fill in the text.
    
    Params
    ------
    masked_index
        Index to mask. If not passed
    
    
    '''
    
    # check arguments
    if masked_index is None and masked_str is None:
        raise ValueError('Must pass either an index or a string to mask')
    elif masked_index is not None and masked_str is not None:
        warnings.warn("Both masked_index and masked_str passed - using masked_str")
        masked_index = None
    
    # tokenize the text
    tokenized_text = tokenizer.tokenize(text)

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    if masked_index is None:
        masked_str = masked_str.lower()
        masked_index = tokenized_text.index(masked_str)
    tokenized_text[masked_index] = '[MASK]'
    

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text)
    for i in range(masked_index + 1, len(tokenized_text)): # note this is hardcoded, needs to be changed
        segments_ids[i] = 1
    

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)
    masked_predictor.to(device)

    # Predict all tokens
    with torch.no_grad():
        outputs = masked_predictor(tokens_tensor, token_type_ids=segments_tensors)
        predictions = outputs[0]

    preds_masked = predictions[0, masked_index].softmax(dim=0).detach().cpu().numpy()
    inds_max = np.argsort(preds_masked)[::-1]
    return inds_max, preds_masked