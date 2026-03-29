import numpy as np
import collections
import torch
from torch.autograd import Variable
import rnn

start_token = 'G'
end_token = 'E'

def process_poems1(file_name):
    poems = []
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except:
                pass
    poems = sorted(poems, key=lambda line: len(line))
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words

def to_word(predict, vocabs):
    sample = np.argmax(predict)
    if sample >= len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]

print("="*50)
print("加载数据和模型...")
print("="*50)

# 先加载模型看看实际的vocab size
checkpoint = torch.load('./poem_generator_rnn')
print(f"模型中的vocab_size: {checkpoint['fc.weight'].shape[0]}")

# 使用与训练时相同的参数
vocab_size = checkpoint['fc.weight'].shape[0]

poems_vector, word_int_map, vocabularies = process_poems1('./poems.txt')
print(f"词汇表大小: {len(vocabularies)}")
print(f"诗歌数量: {len(poems_vector)}")

# 使用checkpoint中的vocab_size
word_embedding = rnn.word_embedding(vocab_length=vocab_size, embedding_dim=100)
rnn_model = rnn.RNN_model(batch_sz=64, vocab_len=vocab_size,
                         word_embedding=word_embedding, embedding_dim=100, lstm_hidden_dim=128)

print("加载已训练的模型...")
rnn_model.load_state_dict(checkpoint)
print("模型加载成功!")

def gen_poem(begin_word):
    poem = begin_word
    word = begin_word
    max_length = 50
    for _ in range(max_length):
        input = np.array([word_int_map[w] for w in poem], dtype=np.int64)
        input = Variable(torch.from_numpy(input))
        output = rnn_model(input, is_test=True)
        word = to_word(output.data.tolist()[-1], vocabularies)
        poem += word
        if word == end_token or len(poem) > max_length:
            break
    return poem.replace(start_token, '').replace(end_token, '')

print("\n" + "="*50)
print("开始生成诗歌...")
print("="*50)

begin_words = ['日', '红', '山', '夜', '湖', '海', '月']

for word in begin_words:
    poem = gen_poem(word)
    print(f"\n以'{word}'开头的诗:")
    print(poem)
    print("-" * 30)

print("\n" + "="*50)
print("诗歌生成完成!")
print("="*50)
