import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from config import get_config

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter

import warnings
from tqdm import tqdm

from pathlib import Path

def greedy_decode(model , source, source_mask, tokenizer_src, tokenizer_tgt, maxlen, device):
    sos_idx = tokenizer_src.token_to_id('[SOS]')
    eos_idx = tokenizer_src.token_to_id('[EOS]')

    #encoder once
    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == maxlen:
            break
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source).to(device)

        out  = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:, -1])

        _, next_word = torch.max(prob, dim = 1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim = 1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, val_loader, tokenizer_src, tokenizer_tgt, maxlen, device, print_msg, writer, num_examples = 2):
    model.eval()
    count = 0

    with torch.no_grad():
        for batch in val_loader: 
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, maxlen,device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            print_msg('-' * 100)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                break


def get_all_seq(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang)) # putting eng or it in format in the string
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace() # split the sentence by whitespace
        trainer = WordLevelTrainer(special_tokens = ['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_seq(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_ds(config):# this method has and operates on the real data
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split = 'train')

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])


    #90 and 10% split of data
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")   

    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def train_model(config):
    #device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    Path(config['model_folder']).mkdir(parents = True, exist_ok =True)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config['seq_len'], config['d_model']).to(device)

    #tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)

    
    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id('[PAD]'), label_smoothing = 0.1).to(device)


    for epoch in range(config['num_epochs']):
        
        batch_iterator = tqdm(train_dataloader, desc = f'Processing epoch {epoch}')

        for data in batch_iterator: # for batch_of_data in train_dataloader
            model.train()
            encoder_input = data['encoder_input'].to(device) # batch, seq_len
            decoder_input = data['decoder_input'].to(device) # batch_size, seq_len
            encoder_mask = data['encoder_mask'].to(device)  # batch_size, 1, 1, seq_len
            decoder_mask = data['decoder_mask'].to(device) # batch_size, 1, seq_len, seq_len

            #run through the model

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output) # batch, seq_len, tgt_vocab_size

            label = data['label'].to(device) # batch_size, seq_len

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)) 

            batch_iterator.set_postfix({f'loss': f"{loss.item():6.3f}"})

            #backprop
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), writer)




if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)