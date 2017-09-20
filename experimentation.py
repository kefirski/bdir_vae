from blocks.inference_block import InferenceBlock
import torch as t

if __name__ == '__main__':

    x = t.FloatTensor([[1, 2, 3], [4, 5, 6]])

    print(x.unsqueeze(1).repeat(1, 5, 1).view(-1, 3))