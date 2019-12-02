import abc
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard

from . import misc_utils


class TrainingBuddy:
    def __init__(self, name, model, optimizer_names=["primary"], load_checkpoint=True,
                 log_dir="logs", checkpoint_dir="checkpoints"):
        # CUDA boilerplate
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            model.cuda()
        else:
            self._device = torch.device("cpu")
        print("Using device:", self._device)
        torch.autograd.set_detect_anomaly(True)

        # Training boilerplate
        assert isinstance(model, nn.Module)
        self._name = name
        self._model = model
        self._writer = torch.utils.tensorboard.SummaryWriter(
            log_dir + "/" + name)
        self._checkpoint_dir = checkpoint_dir
        self._steps = 0
        self._log_namespace = None

        # Create optimizers -- we might want to use a different one for each
        # loss function
        self._optimizers = {}
        for name in optimizer_names:
            self._optimizers[name] = optim.Adadelta(self._model.parameters())

        # Load checkpoint using model name
        if load_checkpoint:
            self.load_checkpoint()

    def minimize(self, loss, retain_graph=False,
                 optimizer_name="primary", checkpoint_interval=1000):

        assert optimizer_name in self._optimizers.keys()

        # Take gradient step
        self._optimizers[optimizer_name].zero_grad()
        loss.backward(retain_graph=retain_graph)
        self._optimizers[optimizer_name].step()

        # Update step & checkpoint
        self._steps += 1
        if self._steps % checkpoint_interval == 0:
            self.save_checkpoint()

    def log_namespace(self, namespace):
        # TODO: support nesting?
        class _Namespace:
            def __enter__(unused_self):
                self._log_namespace = namespace
                return unused_self

            def __exit__(*unused):
                self._log_namespace = None
                return

        return _Namespace()

    def log(self, name, value):
        if self._log_namespace is not None:
            name = "{}/{}".format(self._log_namespace, name)
        self._writer.add_scalar(name, value, global_step=self._steps)

    def save_checkpoint(self, path=None):
        if path is None:
            path = "{}/{}-{:016d}.ckpt".format(self._checkpoint_dir,
                                          self._name, self._steps)

        optimizer_states = {}
        for name, optimizer in self._optimizers.items():
            optimizer_states[name] = optimizer.state_dict()

        state = {
            'state_dict': self._model.state_dict(),
            'optimizers': optimizer_states,
            'steps': self._steps
        }
        torch.save(state, path)
        print("Saved checkpoint to path:", path)

    def load_checkpoint(self, path=None):
        if path is None:
            path_choices = glob.glob(
                "{}/{}-*.ckpt".format(self._checkpoint_dir, self._name))
            if len(path_choices) == 0:
                print("No checkpoint found")
                return
            steps = []
            for choice in path_choices:
                prefix_len = len(
                    "{}/{}-".format(self._checkpoint_dir, self._name))
                suffix_len = len(".ckpt")
                string_steps = choice[prefix_len:-suffix_len]
                steps.append(int(string_steps))

            path = path_choices[np.argmax(steps)]
            expected_steps = np.max(steps)

            state = torch.load(path)
            assert state['steps'] == np.max(steps)
        else:
            state = torch.load(path)

        self._model.load_state_dict(state['state_dict'])

        for name, state_dict in state['optimizers'].items():
            self._optimizers[name].load_state_dict(state_dict)

        self._steps = state['steps']

        print("Loaded checkpoint from path:", path)


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
            output[key] = to_numpy(value)
    elif type(x) in (list, tuple):
        # Convert lists of values
        output = []
        for value in x:
            output.append(to_numpy(value))
    else:
        assert False, "Invalid datatype {}!".format(type(x))

    return output
