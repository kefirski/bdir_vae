from blocks.inference_block import InferenceBlock

if __name__ == '__main__':

    inf = InferenceBlock(input=10, mu=12, std=13)
    print(inf.iaf)