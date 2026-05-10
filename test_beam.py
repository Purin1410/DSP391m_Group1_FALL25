import torch
from torch.nn import functional as F

batch_size = 4
beam_size = 5
max_len = 10
vocab_size = 20
SOS = 1
EOS = 2
PAD = 0

input_ids = torch.zeros(batch_size, 1, dtype=torch.long)
input_ids[:2, 0] = SOS
input_ids[2:, 0] = EOS

end_tokens = torch.zeros(batch_size, dtype=torch.long)
end_tokens[:2] = EOS
end_tokens[2:] = SOS

beam_scores = torch.zeros(batch_size, dtype=torch.float)

for step in range(max_len):
    # Dummy logits
    logits = torch.randn(input_ids.shape[0], vocab_size)
    logprobs = F.log_softmax(logits, dim=-1)
    
    if step > 0:
        end_tokens_expanded = end_tokens.unsqueeze(1).expand(batch_size, beam_size).reshape(-1)
        is_done = (input_ids == end_tokens_expanded.unsqueeze(1)).any(dim=1)
        
        logprobs[is_done, :] = -float('inf')
        logprobs[is_done, PAD] = 0.0
    
    next_scores = beam_scores.unsqueeze(-1) + logprobs
    
    if step == 0:
        next_scores = next_scores.view(batch_size, vocab_size)
        beam_scores, next_tokens = torch.topk(next_scores, beam_size, dim=1)
        input_ids = input_ids.repeat_interleave(beam_size, dim=0)
        end_tokens = end_tokens.repeat_interleave(beam_size, dim=0)
        input_ids = torch.cat([input_ids, next_tokens.view(-1, 1)], dim=1)
        beam_scores = beam_scores.view(-1)
    else:
        next_scores = next_scores.view(batch_size, beam_size * vocab_size)
        beam_scores, next_tokens = torch.topk(next_scores, beam_size, dim=1)
        
        beam_indices = next_tokens // vocab_size
        token_indices = next_tokens % vocab_size
        
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, beam_size)
        flat_indices = batch_indices * beam_size + beam_indices
        flat_indices = flat_indices.view(-1)
        
        input_ids = input_ids[flat_indices]
        input_ids = torch.cat([input_ids, token_indices.view(-1, 1)], dim=1)
        beam_scores = beam_scores.view(-1)

print(input_ids)
print(beam_scores)
