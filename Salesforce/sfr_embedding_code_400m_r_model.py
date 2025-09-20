import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torch


def embedding_retreival(query,docs):
       
      passages = [d.page_content for d in docs]

      tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Code-400M_R', trust_remote_code=True)
      model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Code-400M_R', trust_remote_code=True)

      # Encode query
      q_inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=8192)
      q_outputs = model(**q_inputs)

      # We use mean pooling for the best Result
      q_emb = q_outputs.last_hidden_state.mean(dim=1)   # mean pooling
      q_emb = F.normalize(q_emb, p=2, dim=1)

      # Encode passages
      p_inputs = tokenizer(passages, return_tensors='pt', padding=True, truncation=True, max_length=8192)
      p_outputs = model(**p_inputs)

      # We use mean pooling for the best Result
      p_emb = p_outputs.last_hidden_state.mean(dim=1)
      p_emb = F.normalize(p_emb, p=2, dim=1)

      # Similarity scores
      scores = (q_emb @ p_emb.T) * 100
      print(scores.tolist())

      # Top k similarity
      top_k = min(3, scores.shape[1])
      topk_scores, topk_indices = torch.topk(scores, k=top_k, dim=1)

      top_k_retreival = [{ i+1: passages[topk_indices[0][i]] , "score" : topk_scores[0][i].item() } for i in range(top_k)]

      top_k_retreival
      return top_k_retreival