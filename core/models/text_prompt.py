# -*- encoding: utf-8 -*-
"""
@File    :   text_prompt.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/20 10:27 AM   Gzhlaker      1.0         None
"""
import torch
from rich.traceback import install
from rich.progress import track

from core.manager.printer import Printer
from three import clip


class TextPrompt:
    """

    """

    def __init__(self):
        self.text_templates = [
            f"a photo of action {{}}",
            f"a picture of action {{}}",
            f"Human action of {{}}",
            f"{{}}, an action",
            f"{{}} this is an action",
            f"{{}}, a video of action",
            f"Playing action of {{}}",
            f"{{}}",
            f"Playing a kind of action, {{}}",
            f"Doing a kind of action, {{}}",
            f"Look, the human is {{}}",
            f"Can you recognize the action of {{}}?",
            f"Video classification of {{}}",
            f"A video of {{}}",
            f"The man is {{}}",
            f"The woman is {{}}"
        ]

    def __call__(self, data):
        """

        Parameters
        ----------
        self
        data

        Returns
        -------

        """
        _text_token_dict = {}
        _num_text_template = len(self.text_templates)
        for _text_template_index, _text_template in enumerate(self.text_templates):
            Printer.print_log("create the {} prompt".format(_text_template_index))
            _text_token_list = []
            for i in track(range(len(data.sentences)), description="create prompt {}".format(_text_template)):
                _text_token_list.append(clip.tokenize(_text_template.format(data.sentences[list(data.sentences.keys())[i]]), truncate=True))
            _text_token_dict[_text_template_index] = torch.cat(_text_token_list)
        # for ii, txt in enumerate(self.text_templates):
        #     _text_token_dict[ii] = torch.cat(
        #         [clip.tokenize(txt.format(c)) for i, c in data.classes]
        #     )
        _classes = torch.cat([value for key, value in _text_token_dict.items()])
        return _classes, _num_text_template, _text_token_dict
