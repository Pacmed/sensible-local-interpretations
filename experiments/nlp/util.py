import torch
import numpy as np

def pred(s: str, tokenizer, clf, device='cpu'):
    '''Predict for a string given tokenizer and (returns class 1 - the positive class)
    '''
    with torch.no_grad():
        # print('encoded', tokenizer.convert_ids_to_tokens(tokenizer.encode(s)))
        input_ids = torch.tensor(tokenizer.encode(s), device=device).unsqueeze(0)  # Batch size 1
        pred = clf(input_ids)[0].detach().cpu().softmax(dim=1).numpy().flatten()
    return pred[1]


def predict_masked(masked_predictor, tokenizer,
                   text="[CLS] This great movie was very good . [SEP] I thoroughly enjoyed it [SEP]",
                   masked_index=6,
                   device='cpu'):
    tokenized_text = tokenizer.tokenize(text)

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    tokenized_text[masked_index] = '[MASK]'
    assert tokenized_text == ['[CLS]', 'this', 'great', 'movie', 'was', 'very', '[MASK]', '.', 
                              '[SEP]', 'i', 'thoroughly', 'enjoyed', 'it', '[SEP]']

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