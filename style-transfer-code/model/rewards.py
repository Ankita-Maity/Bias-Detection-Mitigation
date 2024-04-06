import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity, kl_div
#from sklearn.metrics.pairwise import cosine_similarity
import re
from icecream import ic

def embedding_reward(ref_texts, generated_texts, emb_device, embmodel, embtok):
    sentences = ref_texts + generated_texts
    tokens = embtok(sentences, truncation=True, max_length = 250, padding='max_length', 
        return_tensors='pt').to(emb_device)

    with torch.no_grad():
        outputs = embmodel(**tokens)

    #embeddings = torch.mean(torch.stack(outputs.hidden_states[-2:]), dim=0)
    embeddings = outputs.last_hidden_state
    attention_mask = tokens['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask

    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    bs = mean_pooled.shape[0]
    bs_h = bs // 2
    cosine_sim = cosine_similarity(mean_pooled[:bs_h], mean_pooled[bs_h:])
    
    return cosine_sim

def remove_extra_ids(sents_list):
    pattern = re.compile(r'<extra_id_\d+>।?')
    return list(map(lambda s: pattern.sub('', s), sents_list))

#returns a value between 0 and 1 based on how many words in the generated text are biased
#the biased words are assumed to be words which were removed from the input_text in order to produce the reference text
def unchanged_reward(input_text_list, reference_text_list, generated_text_list, embtok):
    unchanged_reward_list = []
    generated_text_list = remove_extra_ids(generated_text_list)

    for input_text, reference_text, generated_text in zip(input_text_list, reference_text_list, generated_text_list):
        input_tokens = set(embtok.tokenize(input_text))
        reference_tokens = set(embtok.tokenize(reference_text))
        generated_tokens = embtok.tokenize(generated_text)

        biased_tokens = input_tokens.difference(reference_tokens)
        
        generated_tokens_count = len(generated_tokens)
        unbiased_count = sum(1 for word in generated_tokens if word not in biased_tokens)
        reward = unbiased_count / float(generated_tokens_count) if generated_tokens_count > 0 else 0.0
        
        unchanged_reward_list.append(reward)

    unchanged_reward = torch.FloatTensor(unchanged_reward_list)
    return unchanged_reward


def kl_reward(ref_texts, generated_texts, clf_device, clf_model, clf_tok):
    sentences = ref_texts + generated_texts
    tokens = clf_tok(sentences, truncation=True, max_length=250, padding='max_length', 
        return_tensors='pt', return_token_type_ids=False).to(clf_device)
    
    with torch.no_grad():
        clf_pred = clf_model(**tokens)
    
    clf_pred = torch.log(torch.clamp(clf_pred, min=1e-9))
    bs = clf_pred.shape[0]
    bs_h = bs // 2
    ref_pred = clf_pred[:bs_h]
    gen_pred = clf_pred[bs_h:]
    
    kls = torch.zeros(bs_h).to(clf_device)
    max_val = 10.3616
    for i in range(bs_h):
        kl = (max_val - kl_div(input=gen_pred[i, :], target=ref_pred[i, :], 
            reduction='batchmean', log_target=True)) / max_val
        kls[i] = kl.item()
    kls = torch.pow(kls, 4)

    return kls