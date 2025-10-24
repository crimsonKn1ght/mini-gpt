# imports
import torch

from model import BigramLanguageModel
from eval import estimate_loss
from batch import get_batch

from config import max_iters, eval_interval, learning_rate, device
from read_file import decode
# --------------

torch.manual_seed(1337)

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(''.join(decode(m.generate(context, max_new_tokens=500)[0].tolist())))
