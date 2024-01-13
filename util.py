import torch

def encode(input_str, encode_func):
    return torch.tensor([encode_func(char) for char in input_str], dtype=torch.long)


def decode(digit_list, decode_func):
    return ''.join([decode_func(digit) for digit in digit_list])


def get_text_coding_func(raw_input):
    chars = sorted(list(set(raw_input)))
    encode_map = {char: i for i, char in enumerate(chars)}
    decode_map = {i: char for i, char in enumerate(chars)}
    return lambda x: encode_map[x], lambda x: decode_map[x]


def get_batch(raw_input, batch_size, timestep):
    # hint: use torch.stack
    idx = torch.randint(len(raw_input)-timestep-1, (batch_size, ))#.long()
    x = torch.stack([raw_input[i:i+timestep] for i in idx])
    y = torch.stack([raw_input[i+1:i+timestep+1] for i in idx])
    return x, y


# with open('input.txt', 'r') as f:
#     input_text = f.read()

# ef, df = get_text_coding_func(input_text)
# raw_input = encode(input_text[:200], ef)
# # print(raw_input)
# print(get_batch(raw_input, 3, 8))
