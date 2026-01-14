import torch
import torch.nn as nn
from src.modules.attend import Attend
from src.modules.re_attend import ReAttend
from src.modules.combine import Combine
from src.modules.classify import Classify
from src.modules.measure import Measure
from src.layout.parser_to_layout import parse_question_to_layout

class NMNPipeline(nn.Module):
    def __init__(self, image_dim, num_classes_dict, use_lstm=False, lstm_module=None):
        super().__init__()
        self.attend = Attend(image_dim)
        self.re_attend = ReAttend(image_dim)
        self.combine = Combine()
        self.classify_modules = nn.ModuleDict({
            k: Classify(image_dim, v) for k,v in num_classes_dict.items()
        })
        self.measure_modules = nn.ModuleDict({
            "is": Measure(image_dim, 2),  
        })
        self.use_lstm = use_lstm
        self.lstm_module = lstm_module

    def forward(self, image_features, question):
        layout = parse_question_to_layout(question)
        out = self._execute_module(layout, image_features)
        if self.use_lstm and self.lstm_module:
            lstm_out = self.lstm_module(question)
            out = (torch.softmax(out, dim=-1) * torch.softmax(lstm_out, dim=-1)) ** 0.5
        return out

    def _execute_module(self, node, image_features):
        if node.type == "attend":
            return self.attend(image_features)
        elif node.type == "re_attend":
            child_out = self._execute_module(node.args[0], image_features)
            return self.re_attend(child_out)
        elif node.type == "combine":
            out1 = self._execute_module(node.args[0], image_features)
            out2 = self._execute_module(node.args[1], image_features)
            self.combine.mode = node.instance
            return self.combine(out1, out2)
        elif node.type == "classify":
            child_out = self._execute_module(node.args[0], image_features)
            return self.classify_modules[node.instance](image_features, child_out)
        elif node.type == "measure":
            child_out = self._execute_module(node.args[0], image_features)
            return self.measure_modules[node.instance](child_out)
        else:
            raise ValueError(f"Unknown module type {node.type}")
