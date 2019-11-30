import abc
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard

class TrainingBuddy:
    def __init__(self, name, model, load_checkpoint=True):
        # CUDA boilerplate
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            model.cuda()
        else:
            self._device = torch.device("cpu")
        print("Using device:", self._device)
        torch.autograd.set_detect_anomaly(True)

        # Misc stuff
        assert isinstance(model, nn.Module)
        self._name = name
        self._model = model
        self._writer = torch.utils.tensorboard.SummaryWriter("logs/" + name)
        self._optimizer = optim.Adadelta(self._model.parameters())
        self._steps = 0

        if load_checkpoint:
            self.load_checkpoint()

    def minimize(self, loss, retain_graph=False):
        self._optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self._optimizer.step()

        self._steps += 1

    def log(self, name, value):
        self._writer.add_scalar(name, value, global_step=self._steps)

    def save_checkpoint(self, path=None):
        if path is None:
            path = "checkpoints/{}-{}.ckpt".format(self._name, self._steps)

        state = {
            'state_dict': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'steps': self._steps
        }
        torch.save(state, path)
        print("Saved checkpoint to path:", path)

    def load_checkpoint(self, path=None):
        if path is None:
            path_choices = glob.glob(
                "checkpoints/{}-*.ckpt".format(self._name))
            if len(path_choices) == 0:
                print("No checkpoint found")
                return
            steps = []
            for choice in path_choices:
                prefix_len = len("checkpoints/{}-".format(self._name))
                suffix_len = len(".ckpt")
                string_steps = choice[prefix_len:-suffix_len]
                steps.append(int(string_steps))
                assert str(steps[-1]) == string_steps

            path = path_choices[np.argmax(steps)]
            expected_steps = np.max(steps)

            state = torch.load(path)
            assert state['steps'] == np.max(steps)
        else:
            state = torch.load(path)

        self._model.load_state_dict(state['state_dict'])
        self._optimizer.load_state_dict(state['optimizer'])
        self._steps = state['steps']

        print("Loaded checkpoint from path:", path)


class DictIndexWrapper:
    def __init__(self, data):
        assert type(data) == dict

        # Every value in the dict should have the same length
        self._length = None
        for value in data.values():
            length = len(value)
            if self._length is None:
                self._length = length
            else:
                assert length == self._length

        self._data = data

    def __getitem__(self, key):
        output = {}
        for data_key, data_value in self._data.items():
            output[data_key] = data_value[key]
        return output

    def __len__(self):
        return self._length

    def append(self, other):
        for key, value in other.items():
            if key in self._data.keys():
                self._data[key].append(value)
            else:
                self._data[key] = [value]

    def convert_to_numpy(self):
        for key, value in self._data.items():
            self._data[key] = np.asarray(value)


def to_device(x, device, detach=True):
    if type(x) == torch.Tensor:
        # Convert plain arrays
        if detach:
            x = x.detach()
        output = x.to(device)
    elif type(x) == dict:
        # Convert dictionaries of values
        output = {}
        for key, value in x.items():
            output[key] = to_device(value, device, detach)
    elif type(x) in (list, tuple):
        # Convert lists of values
        output = []
        for value in x:
            output.append(to_device(value, device, detach))
    else:
        assert False, "Invalid datatype {}!".format(type(x))
    return output


def to_torch(x, device='cpu'):

    if type(x) == np.ndarray:
        # Convert plain arrays
        output = torch.from_numpy(x).float().to(device)
    elif type(x) == dict:
        # Convert dictionaries of values
        output = {}
        for key, value in x.items():
            output[key] = to_torch(value, device)
    elif type(x) in (list, tuple):
        # Convert lists of values
        output = []
        for value in x:
            output.append(to_torch(value, device))
    else:
        assert False, "Invalid datatype {}!".format(type(x))

    return output


def to_numpy(x):
    if type(x) == torch.Tensor:
        # Convert plain tensors
        output = x.detach().cpu().numpy()
    elif type(x) == dict:
        # Convert dictionaries of values
        output = {}
        for key, value in x.items():
            output[key] = from_torch(value)
    elif type(x) in (list, tuple):
        # Convert lists of values
        output = []
        for value in x:
            output.append(from_torch(value))
    else:
        assert False, "Invalid datatype {}!".format(type(x))

    return output
