import pandas as pd
import pickle as pkl
import numpy as np
from tqdm import tqdm

from pytorch_transformers import BertTokenizer, BertForMaskedLM, BertForSequenceClassification, BertConfig
from util import *
from lime.lime_text import LimeTextExplainer


def calc_contrib_score(word: str, review: str, tokenizer, clf, masked_predictor, device='cuda'):
    '''Get contribution score for (the first occurence of) word in review
    Later: uses masked_predictor to get conditional distr for word
    
    Returns
    -------
    contrib_score_iid, contrib_score_conditional, score_remove
    '''
    # original prediction
    pred_orig = pred(review, tokenizer, clf, device)
    pred_remove = pred(review.replace(word, ''), tokenizer, clf, device)
    
    # use masked predictor to get conditional distr for word
    inds_max, preds_masked = predict_masked(masked_predictor,
                                            tokenizer,
                                            text=review,
                                            masked_str=word)
    
    # conditional mean pred
    n = 200 # there are 30522, we are only going to use the top ones
    # the top ones have the highest prob, so should be okay
    vals = np.zeros(n)
    weights = np.zeros(n)

    for i in range(n):
        ind_max = inds_max[i]
        sampled_word = tokenizer.convert_ids_to_tokens([ind_max])[0]
        s = review.replace(word, sampled_word)
        vals[i] = pred(s, tokenizer, clf, device)
        weights[i] = preds_masked[ind_max]

    ave = np.mean(vals)
    ave_weighted, weights_sum = np.average(vals, weights=weights, returned=True)
    
    # scale up ave_weighted
    ave_weighted /= weights_sum

    return pred_orig - ave, pred_orig - ave_weighted, pred_orig - pred_remove


def s_to_pred(x: list):
    '''Needs to take a list and return 2d-array of probs
    
    Returns
    -------
    probs: np.ndarray (n x num_classes)
        2D-array of probs 
    '''
    if not type(x) is list:
        print('arg must be a list!')
    if type(x) is list:
        results = []
        for s in x:
            if all([c.isspace() for c in s]):
                results.append(np.array([0.5, 0.5]))
            else:
                input_ids = torch.tensor(tokenizer.encode(s), device=device).unsqueeze(0)  # Batch size 1
                results.append(clf.forward(input_ids)[0].detach().cpu().numpy().flatten())
        return np.array(results).reshape(-1, 2)

print('loading models and data...')
default = 'bert-base-uncased'
mdir = '/scratch/users/vision/chandan/pacmed/glue/SST-2-3epoch' # '/scratch/users/vision/chandan/pacmed/glue/SST-2-middle/'
device = 'cpu'

tokenizer = BertTokenizer.from_pretrained(mdir)
clf = BertForSequenceClassification.from_pretrained(mdir).eval().to(device)
masked_predictor = BertForMaskedLM.from_pretrained(default).eval().to(device)

lines = open('data/stsa.binary.test', 'r').read()
lines = [line for line in lines.split('\n') if not line is '']
classes = [int(line[0]) for line in lines]
reviews = [line[2:] for line in lines]


num_reviews = 1821 # 1821
save_freq = 1
scores_iid = {}
scores_conditional = {}
scores_remove = {}
scores_lime = {}

# loop over reviews
print('looping over dset...')
for i in tqdm(range(num_reviews)):
    review = reviews[i]
    
    # loop over words in review
    '''
    for word in review.split():
        
        if not word in [';', ',', '.', '!', '?', '/']:

            # get contrib score for word
            try:
                score_iid, score_conditional, score_remove = \
                calc_contrib_score(word, review, tokenizer, clf, masked_predictor, device)

                if not word in scores_iid:
                    scores_iid[word] = [score_iid]
                    scores_conditional[word] = [score_conditional]
                    scores_remove[word] = [score_remove]
                else:
                    scores_iid[word].append(score_iid)
                    scores_conditional[word].append(score_conditional)
                    scores_remove[word].append(score_remove)
            except:
                pass
    '''
    
    # get lime scores
    try:
        explainer = LimeTextExplainer(class_names=['pos', 'neg'])
        exp = explainer.explain_instance(review, s_to_pred)
        exp_list = exp.as_list()
        for word, score_lime in exp_list:
            if not word in scores_lime:
                scores_lime[word] = [score_lime]
            else:
                scores_lime[word].append(score_lime)
    except:
        pass

            
    if i % save_freq == 0:
        pkl.dump({'iid': scores_iid, 'conditional': scores_conditional, 
                  'remove': scores_remove, 'lime': scores_lime},
                 open(f'scores_lime_{i}.pkl', 'wb'))