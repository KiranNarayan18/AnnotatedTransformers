import torch

from models.model import make_model
from models.attention import subsequent_mask

def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()

    src = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)
    src_mask = torch.ones(1,1,10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1,1).type_as(src)

    for i in range(9):
        out = test_model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src))

        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]

        ys = torch.cat([ys, torch.ones(1,1).type_as(src).fill_(next_word)], dim=1)

    print("Example Untrained Model Prediction:", ys)


def run_tests():    
    for _ in range(1):
        print("Running inference test...")
        inference_test()

RUN_EXAMPLES = True
def show_example(fn, args=[]):
  if __name__=="__main__" and RUN_EXAMPLES:
    return fn(*args)


show_example(run_tests)
