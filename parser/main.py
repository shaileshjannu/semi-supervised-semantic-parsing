from model import Model
from preprocess import load_data

train_path = r"..\data\orig\geo880-train.examples"
val_path = r"..\data\orig\geo880-test.examples"
questions_lang, targets_lang, pairs = load_data(train_path)
_, _, val_pairs = load_data(val_path)

model = Model(questions_lang, targets_lang, 100, 2, 0, 0.001, r'.\log\bidirectional',
              teacher_forcing_ratio=0.5, clip=5.0, cuda=True)
model.train(pairs, val_pairs, 200, eval_freq=1000)