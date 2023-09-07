import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# F.max_pool2d needs kernel_size and stride. If only one argument is passed, 
# then kernel_size = stride

from .audio import MelspectrogramStretch
# from torchparse import parse_cfg

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self, config=''):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.classes = None
        
    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + '\nTrainable parameters: {}'.format(params)
        # print(super(BaseModel, self))


# Architecture inspiration from: https://github.com/keunwoochoi/music-auto_tagging-keras
class AudioCRNN(BaseModel):
    def __init__(self, classes=10, config={}, state_dict=None):
        super(AudioCRNN, self).__init__(config)
        
        in_chan = 1

        self.classes = classes
        self.lstm_units = 64
        self.lstm_layers = 2
        self.spec = MelspectrogramStretch(hop_length=None, 
                                num_mels=128, 
                                fft_length=2048, 
                                norm='whiten', 
                                stretch_param=[0.4, 0.4])

        # shape -> (channel, freq, token_time)
        # self.net = parse_cfg(config['cfg'], in_shape=[in_chan, self.spec.n_mels, 400])
        self.net = nn.ModuleDict({
            'convs': nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
                nn.BatchNorm2d(32),
                nn.ELU(alpha=1.0),
                nn.MaxPool2d(kernel_size=3, stride=3),
                nn.Dropout(p=0.1),
                nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
                nn.BatchNorm2d(64),
                nn.ELU(alpha=1.0),
                nn.MaxPool2d(kernel_size=4, stride=4),
                nn.Dropout(p=0.1),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
                nn.BatchNorm2d(64),
                nn.ELU(alpha=1.0),
                nn.MaxPool2d(kernel_size=4, stride=4),
                nn.Dropout(p=0.1)
            ),
            'recur': nn.LSTM(128, 64, num_layers=2, batch_first=True),
            'dense': nn.Sequential(
                nn.Dropout(p=0.3),
                nn.BatchNorm1d(64),
                nn.Linear(in_features=64, out_features=self.classes)
            )
        })

    def _many_to_one(self, t, lengths):
        return t[torch.arange(t.size(0)), lengths - 1]

    def modify_lengths(self, lengths):
        def safe_param(elem):
            return elem if isinstance(elem, int) else elem[0]
        
        for name, layer in self.net['convs'].named_children():
            #if name.startswith(('conv2d','maxpool2d')):
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
                p, k, s = map(safe_param, [layer.padding, layer.kernel_size,layer.stride]) 
                lengths = ((lengths + 2*p - k)//s + 1).long()

        return torch.where(lengths > 0, lengths, torch.tensor(1, device=lengths.device))
    

    def forward(self, batch):    
        # x-> (batch, time, channel)
        x, lengths, _ = batch # unpacking seqs, lengths and srs
        # x-> (batch, channel, time)
        xt = x.float().transpose(1,2)
        # xt -> (batch, channel, freq, time)
        xt, lengths = self.spec(xt, lengths)                

        # (batch, channel, freq, time)
        xt = self.net['convs'](xt)
        lengths = self.modify_lengths(lengths)

        # xt -> (batch, time, freq, channel)
        x = xt.transpose(1, -1)

        # xt -> (batch, time, channel*freq)
        batch, time = x.size()[:2]
        x = x.reshape(batch, time, -1)
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
    
        # x -> (batch, time, lstm_out)
        x_pack, hidden = self.net['recur'](x_pack)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)
        
        # (batch, lstm_out)
        x = self._many_to_one(x, lengths)
        # (batch, classes)
        x = self.net['dense'](x)

        x = F.log_softmax(x, dim=1)

        return x

    def predict(self, x):
        with torch.no_grad():
            out_raw = self.forward( x )
            out = torch.exp(out_raw)
            max_ind = out.argmax().item()        
            return self.classes[max_ind], out[:,max_ind].item()


class AudioCNN(AudioCRNN):

    def forward(self, batch):
        x, _, _ = batch
        # x-> (batch, channel, time)
        x = x.float().transpose(1,2)
        # x -> (batch, channel, freq, time)
        x = self.spec(x)                

        # (batch, channel, freq, time)
        x = self.net['convs'](x)

        # x -> (batch, time*freq*channel)
        x = x.view(x.size(0), -1)
        # (batch, classes)
        x = self.net['dense'](x)

        x = F.log_softmax(x, dim=1)

        return x


class AudioRNN(AudioCRNN):

    def forward(self, batch):    
        # x-> (batch, time, channel)
        x, lengths, _ = batch # unpacking seqs, lengths and srs

        # x-> (batch, channel, time)
        x = x.float().transpose(1,2)
        # x -> (batch, channel, freq, time)
        x, lengths = self.spec(x, lengths)                

        # x -> (batch, time, freq, channel)
        x = x.transpose(1, -1)

        # x -> (batch, time, channel*freq)
        batch, time = x.size()[:2]
        x = x.reshape(batch, time, -1)
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
    
        # x -> (batch, time, lstm_out)
        x_pack, hidden = self.net['recur'](x_pack)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)
        
        # (batch, lstm_out)
        x = self._many_to_one(x, lengths)
        # (batch, classes)
        x = self.net['dense'](x)

        x = F.log_softmax(x, dim=1)

        return x
