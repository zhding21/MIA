import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
# from transformers import BertTokenizer, BertModel
from functools import partial
from torch.utils.data import DataLoader
import torch.nn.functional as F

from load_data import load_dataset, my_collate
# from model import Transformer_CNN_RNN_Attention


class Transformer_CNN_RNN_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('./word_embedding')
        self.base_model = AutoModel.from_pretrained('./word_embedding')
        for param in self.base_model.parameters():
            param.requires_grad = (True)

        # Define the hyperparameters
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 100

        # TextCNN
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                       kernel_size=(K, self.base_model.config.hidden_size)) for K in self.filter_sizes])

        # LSTM
        self.lstm = nn.LSTM(input_size=self.base_model.config.hidden_size,
                            hidden_size=512,
                            num_layers=1,
                            batch_first=True)
        # Self-Attention
        self.key_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.query_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.value_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self._norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)

        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(812, 128),
            nn.Linear(128, 32),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )

    def conv_pool(self, tokens, conv):
        # x -> [batch,1,text_length,768]
        tokens = conv(tokens)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1, 1]
        tokens = F.relu(tokens)
        tokens = tokens.squeeze(3)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1]
        tokens = F.max_pool1d(tokens, tokens.size(2))  # shape[batch, out_channels, 1]
        out = tokens.squeeze(2)  # shape[batch, out_channels]
        return out

    def forward(self, inputs):
        # raw_outputs = self.base_model(**inputs)
        raw_outputs = self.base_model(inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'])
        tokens = raw_outputs.last_hidden_state
        K = self.key_layer(tokens)
        Q = self.query_layer(tokens)
        V = self.value_layer(tokens)
        attention = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact)
        attention_output = torch.bmm(attention, V)
        cnn_tokens = attention_output.unsqueeze(1)
        cnn_out = torch.cat([self.conv_pool(cnn_tokens, conv) for conv in self.convs], 1)
        rnn_tokens = tokens
        rnn_outputs, _ = self.lstm(rnn_tokens)
        rnn_out = rnn_outputs[:, -1, :]
        out = torch.cat((cnn_out, rnn_out), 1)
        predicts = self.block(out)
        return predicts

    def data_load(self, msg):

        return 1



class Config:
    def __init__(self, path=None, cfg=None):
        self.__data = {}
        self.__data = cfg if cfg is not None else {}
        if path is not None and cfg is None:
            with open(path, 'rb') as default_config:
                content = self.load_yaml(default_config)
                self.__data = self.update_config(self.__data, content)

    def load_yaml(self, file):
        try:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        except AttributeError:
            conf_dict = yaml.load(file)
        return conf_dict

    def update_config(self, conf, new_conf):
        for item in new_conf.keys():
            if type(new_conf[item]) == dict and item in conf.keys():
                conf[item] = self.update_config(conf[item], new_conf[item])
            else:
                conf[item] = new_conf[item]
        return conf

    def __getattr__(self, item):
        if type(self.__data[item]) == dict:
            return Config(cfg=self.__data[item])
        return self.__data[item]

    def __getitem__(self, item):
        return self.__data[item]


def main():
    config = Config(path='data/Config.yaml')
    save_model_path = os.path.join(config.output_dir, 'DLL-Best')
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.embed_dir)
    base_model = AutoModel.from_pretrained(config.embed_dir)
    # base_model = BertModel.from_pretrained('bert-base-uncased')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataloader, test_dataloader = load_dataset(config, tokenizer)

    train_model = Transformer_CNN_RNN_Attention()
    # train_model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_acc, best_model = 0, {}
    for epoch in range(config.epochs):
        print('epochs: {}/{}'.format(epoch + 1, config.epochs))
        train_model.train()
        run_loss, n_train = 0, 0
        with tqdm(train_dataloader, desc='model train') as epoch_iterator:
            for inputs, targets in epoch_iterator:
                optimizer.zero_grad()
                # inputs = {k: v.cuda() for k, v in inputs.items()}
                inputs = {k: v for k, v in inputs.items()}
                # targets = targets.cuda()

                predicts = train_model(inputs)
                loss = criterion(predicts, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                run_loss += loss.item() * targets.size(0)
                n_train += targets.size(0)
        train_loss = run_loss / n_train
        train_model.eval()
        test_loss, n_correct, n_test = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in tqdm(test_dataloader, desc='model test'):
                # inputs = {k: v.cuda() for k, v in inputs.items()}
                inputs = {k: v for k, v in inputs.items()}
                # targets = targets.cuda()
                predicts = train_model(inputs)
                loss = criterion(predicts, targets)

                test_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
                n_test += targets.size(0)
        test_loss = test_loss / n_test
        test_acc = n_correct / n_test
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = {'model_state_dict': train_model.state_dict()}
        print(f'train loss: {train_loss}, test loss: {test_loss}, test acc: {test_acc}.')
    torch.save(best_model, os.path.join(save_model_path, 'checkpoint.pth'))
    print(f'Best acc: {best_acc}')


def load_model(model, model_path, model_name='checkpoint.pth'):
    if not os.path.exists(model_path):
        raise Exception("Model doesn't exists! Train first!")
    model_state_dict = torch.load(os.path.join(model_path, model_name))
    model.load_state_dict(model_state_dict['model_state_dict'])
    print("Model Loaded Success: {}".format(model_path))


def test(data):
    config = Config(path='data/Config.yaml')
    log_info = data.split(' & ')
    message = []
    # log_data, label_id = log_info[:len(log_info) - 1], int(log_info[len(log_info) - 1])
    for log_val in log_info:
        val_list = log_val.replace("'", "").replace('"', '').split(' ')
        for v in val_list:
            if "msg=op=" in v:
                l_h = v.split("msg=op=")[1]
                message.append(l_h)
    msg = '_:_'.join(message)
    msg_tokens = msg.split('_')
    tokenizer = AutoTokenizer.from_pretrained(config.embed_dir)
    base_model = AutoModel.from_pretrained(config.embed_dir)
    # base_model = BertModel.from_pretrained('bert-base-uncased')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    collate_fn = partial(my_collate, tokenizer=tokenizer)
    test_loader = DataLoader([(msg_tokens, -1)], batch_size=config.batch_size, shuffle=True,
                             collate_fn=collate_fn)
    model = Transformer_CNN_RNN_Attention()
    model_state_dict = torch.load(os.path.join(config.output_dir, 'DLL-Best/checkpoint.pth'))
    model.load_state_dict(model_state_dict['model_state_dict'])
    model.cuda()

    # ipt = torch.randn([1, 3, 14]).long()
    # model.eval()
    # trace_model = torch.jit.trace(model, ipt)
    # trace_model.trace(ipt)
    # trace_model.save('TRCRA.pt')
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='model test'):
            inputs = {k: v.cuda() for k, v in inputs.items()}
            # inputs = torch.cat([v for k, v in inputs.items()], dim=0).unsqueeze(dim=0).cuda()
            outputs = model(inputs)
            output = F.softmax(outputs)
            _, pred = torch.max(output, 1)
    pred_label = pred.tolist()
    id_to_label = {1}
    if pred_label[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        print('该用户的操作行为造成了数据滥用！')
    else:
        print('该用户的操作行为没有造成数据滥用！')


if __name__ == '__main__':
    # main()
    msg = "type=USER_CMD msg=audit(1698387482.100:136290): pid=32510 uid=1002 auid=1002 ses=19410 subj=system_u:system_r:user_t:s0-s0:c0.c1023 msg='op=IDENTIFY_STORAGE_LOCATION acct=user exe=/usr/bin/find args=/encrypted_storage hostname=? addr=? terminal=pts/2 res=success' & type=USER_CMD msg=audit(1698387482.200:136291): pid=32511 uid=1002 auid=1002 ses=19411 subj=system_u:system_r:user_t:s0-s0:c0.c1023 msg='op=CRACK_ENCRYPTION_KEY acct=user exe=/usr/bin/crack-tool args=/key_storage' hostname=? addr=? terminal=pts/2 res=success' & type=USER_CMD msg=audit(1698387482.300:136292): pid=32512 uid=1002 auid=1002 ses=19412 subj=system_u:system_r:user_t:s0-s0:c0.c1023 msg='op=DECRYPT_DATA acct='user' exe=/usr/bin/decrypt-tool args=/encrypted_storage/datafile hostname=? addr=? terminal=pts/2 res=success' & 1"
    b = test(msg)
    print(b)
