# -*- encoding: utf-8 -*-
"""
@File    :   train.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/2/20 10:58 AM   Gzhlaker      1.down.sh         None
"""
import os
import sys

import wandb
import numpy
import torch
from torch import optim
from torch.utils.data import DataLoader
from rich.progress import track
from rich.traceback import install
from tqdm import tqdm

from core.augementation.other_augmentation import OtherAugmentation
from core.manager.path_manager import PathManager
from core.manager.printer import Printer
from core.models.image_clip import ImageCLIP
from core.models.text_clip import TextCLIP
from core.models.text_prompt import TextPrompt
from core.models.visual_prompt import VisualPrompt
from core.util.util import gen_label, create_logits, convert_models_to_fp32
from three import clip

install(show_locals=False)
from core.models.learner.warm_up_cosine_annealing_lr import WarmupCosineAnnealingLR
from core.models.learner.warm_up_multi_lr import WarmupMultiStepLR
from core.models.losses.klloss import KLLoss

sys.path.append(".")

from core.run.base_train import BaseTrainer
from core.datasets.how2sign_dataset import How2SignDataset as Dataset


class ActionClipTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()

    def on_user_get_dataset(self):
        if self.config["data"]["dataset"] == "how2sign":
            self._get_how2sign_dataset()
        elif self.config["data"]["dataset"] == "hmdb51":
            self._get_hmdb51_dataset()
        else:
            raise ValueError('Unknown dataset: {}'.format(self.config["data"]["dataset"]))
        return super().on_user_get_dataset()

    def _get_how2sign_dataset(self):
        _transform_train = self._get_train_trans()
        _transform_val = self._get_val_trans()
        self.state["train_dataset"] = Dataset(
            self.config["data"]["train_list"],
            self.config["data"]["label_list"],
            num_segments=self.config["data"]["num_segments"],
            image_tmpl=self.config["data"]["image_tmpl"],
            random_shift=self.config["data"]["random_shift"],
            transform=_transform_train
        )
        self.state["val_dataset"] = Dataset(
            self.config["data"]["val_list"],
            self.config["data"]["label_list"],
            num_segments=self.config["data"]["num_segments"],
            image_tmpl=self.config["data"]["image_tmpl"],
            random_shift=True,
            transform=_transform_val
        )

    def _get_hmdb51_dataset(self):
        pass

    def _get_train_trans(self):
        return OtherAugmentation()(True, self.config)

    def _get_val_trans(self):
        return OtherAugmentation()(False, self.config)

    def on_user_get_dataLoader(self):
        self.state["train_loader"] = DataLoader(
            self.state["train_dataset"],
            batch_size=self.config["data"]["batch_size"],
            num_workers=self.config["data"]["workers"],
            shuffle=True,
            pin_memory=False,
            drop_last=True,
        )
        self.state["val_loader"] = DataLoader(
            self.state["val_dataset"],
            batch_size=self.config["data"]["batch_size"],
            num_workers=self.config["data"]["workers"],
            shuffle=False,
            pin_memory=False,
            drop_last=True,
            # collate_fn=self._get_collate_fn()
        )
        return super().on_user_get_dataLoader()

    def _get_collate_fn(self):
        def collate_fn(batch):
            data = []
            for image, text_id in batch:
                # data.append((image[:24, :, :], text_id))
                if image.size()[0] > 24:
                    Printer.print_log(text_id)
                Printer.print_log(image.size())

            return data

        return collate_fn

    def on_user_get_model(self):
        self.state["model"], self.state["clip_state_dict"] = clip.load(
            self.config["network"]["arch"],
            device=self.state["device"],
            jit=False,
            T=self.config["data"]["num_segments"],
            dropout=self.config["network"]["drop_out"],
            emb_dropout=self.config["network"]["emb_dropout"],
            pretrain=self.config["network"]["init"],
        )
        self.state["fusion_model"] = VisualPrompt(
            self.config["network"]["sim_header"],
            self.state["clip_state_dict"],
            self.config["data"]["num_segments"]
        )
        self.state["model_text"] = TextCLIP(self.state["model"])
        self.state["model_image"] = ImageCLIP(self.state["model"])
        self.state["model_text"] = torch.nn.DataParallel(self.state["model_text"]).cuda()
        self.state["model_image"] = torch.nn.DataParallel(self.state["model_image"]).cuda()
        self.state["fusion_model"] = torch.nn.DataParallel(self.state["fusion_model"]).cuda()
        if self.config["wandb"]:
            wandb.watch(self.state["model"])
            wandb.watch(self.state["fusion_model"])
        return super().on_user_get_model()

    def on_user_get_loss(self):
        Printer.print_log("Image Using KL Loss")
        self.state["loss_img"] = KLLoss()
        Printer.print_log("Text Using KL Loss")
        self.state["loss_txt"] = KLLoss()
        return super().on_user_get_loss()

    def on_user_get_optimizer(self):
        if self.config["solver"]["optim"] == 'adam':
            self._get_adam_optimizer()
        elif self.config["solver"]["optim"] == 'sgd':
            self._get_sgd_optimizer()
        elif self.config["solver"]["optim"] == 'adamw':
            self._get_adamw_optimizer()
        else:
            raise ValueError('Unknown optimizer: {}'.format(self.config["solver"]["optim"]))
        return super().on_user_get_optimizer()

    def _get_adam_optimizer(self):
        self.state["optimizer"] = optim.Adam(
            [
                {'params': self.state["model"].parameters()},
                {
                    'params': self.state["fusion_model"].parameters(),
                    'lr': self.config["solver"]["lr"] * self.config["solver"]["f_ratio"]
                }
            ],
            lr=self.config["solver"]["lr"],
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=0.2
        )  # Params used from paper, the lr is smaller, more safe for fine-tuning to new dataset
        Printer.print_log('Using Adam optimizer')

    def _get_sgd_optimizer(self):
        self.state["optimizer"] = optim.SGD(
            [
                {'params': self.state["model"].parameters()},
                {
                    'params': self.state["fusion_model"].parameters(),
                    'lr': self.config["solver"]["lr"] * self.config["solver"]["f_ratio"]
                }
            ],
            self.config["solver"]["lr"],
            momentum=self.config["solver"]["momentum"],
            weight_decay=self.config["solver"]["weight_decay"]
        )
        Printer.print_log('Using SGD optimizer')

    def _get_adamw_optimizer(self):
        vision_params = list(map(id, self.state["model"].visual.parameters()))
        text_params = filter(lambda p: id(p) not in vision_params,
                             self.state["model"].parameters())

        self.state["optimizer"] = optim.AdamW(
            [
                {'params': text_params},
                {
                    'params': self.state["model"].visual.parameters(),
                    'lr': self.config["solver"]["lr"] * self.config["solver"]["ratio"]
                },
                {
                    'params': self.state["fusion_model"].parameters(),
                    'lr': self.config["solver"]["lr"] * self.config['solver']["f_ratio"]
                }
            ],
            betas=(0.9, 0.98),
            lr=self.config["solver"]["lr"],
            eps=1e-8,
            weight_decay=self.config['solver']["weight_decay"]
        )  # Params used from paper, the lr is smaller, more safe for fine-tuning to new dataset
        for param_group in self.state["optimizer"].param_groups:
            print(param_group['lr'])
        Printer.print_log('Using Adamw optimizer')

    def on_user_get_lr_scheduler(self):
        if self.config["solver"]["type"] == 'cosine':
            self.state["lr_scheduler"] = WarmupCosineAnnealingLR(
                self.state["optimizer"],
                self.config["solver"]["epochs"],
                warmup_epochs=self.config["solver"]["lr_warmup_step"]
            )
        elif self.config["solver"]["type"] == 'multistep':
            if isinstance(self.config["solver"]["lr_decay_step"], list):
                milestones = self.config['solver']["lr_decay_step"]
            elif isinstance(self.config["solver"]["lr_decay_step"], int):
                milestones = [
                    self.config["solver"]["lr_decay_step"] * (i + 1)
                    for i in range(self.config["solver"]["epochs"] //
                                   self.config["solver"]["lr_decay_step"])]
            else:
                raise ValueError(
                    "error learning rate decay step: {}".format(type(self.config["solver"]['lr_decay_step'])))
            self.state["lr_scheduler"] = WarmupMultiStepLR(
                self.state["optimizer"],
                milestones,
                warmup_epochs=self.config["solver"]["lr_warmup_step"]
            )
        else:
            raise ValueError('Unknown lr scheduler: {}'.format(self.config["solver"]["type"]))

    def on_user_get_start_epoch(self):
        self.state["start_epoch"] = self.config["solver"]["start_epoch"]

    def on_user_get_checkpoint(self):

        self.state["classes"], self.state["num_text_aug"], self.state["text_dict"] = TextPrompt()(
            self.state["train_dataset"])

    def on_user_set_grad(self):
        self.state["optimizer"].zero_grad()
        return super().on_user_set_grad()

    def on_user_update_parameter(self):
        return super().on_user_update_parameter()

    def on_user_epoch(self):
        Printer.print_rule("epoch {}".format(self.state["epoch"]), characters="-")
        self.state["model_image"].train()
        self.state["model_text"].train()
        self.state["fusion_model"].train()
        for batch_index, (images, list_id) in enumerate(tqdm(self.state["train_loader"], desc="on epoch {}".format(self.state["epoch"]))):
            # 准备参数
            classes = self.state["classes"]
            num_text_aug = self.state["num_text_aug"]
            text_dict = self.state["text_dict"]

            # 梯度清零
            self.state["optimizer"].zero_grad()

            # 正向传播

            images = images.view((-1, self.config["data"]["num_segments"], 3) + images.size()[-2:])
            b, t, c, h, w = images.size()
            text_id = numpy.random.randint(num_text_aug, size=len(list_id))
            texts = torch.stack([text_dict[j][i, :] for i, j in zip(range(len(list_id)), text_id)])

            images = images.to(self.state["device"]).view(-1, c, h,
                                                          w)  # omit the Image.fromarray if the images already in PIL
            # format, change this line to images=list_image if using preprocess inside the dataset class
            texts = texts.to(self.state["device"])

            image_embedding = self.state["model_image"](images)
            image_embedding = image_embedding.view(b, t, -1)
            image_embedding = self.state["fusion_model"](image_embedding)

            text_embedding = self.state["model_text"](texts)

            if self.config["network"]["fix_text"]:
                text_embedding.detach_()

            logit_scale = self.state["model"].logit_scale.exp()
            logits_per_image, logits_per_text = create_logits(image_embedding, text_embedding, logit_scale)

            ground_truth = torch.tensor(gen_label(list_id), dtype=image_embedding.dtype, device=self.state["device"])

            # 计算损失
            loss_images = self.state["loss_img"](logits_per_image, ground_truth)
            loss_texts = self.state["loss_txt"](logits_per_text, ground_truth)
            total_loss = (loss_images + loss_texts) / 2
            if self.config["wandb"]:
                wandb.log({"train_total_loss": total_loss})
                wandb.log({"train_loss_imgs": loss_images})
                wandb.log({"train_loss_texts": loss_texts})
                wandb.log({"lr": self.state["optimizer"].param_groups[0]['lr']})

            # 反向传播
            total_loss.backward()

            # 更新权重
            if self.state["device"] == "cpu":
                self.state["optimizer"].step()
            else:
                convert_models_to_fp32(self.state["model"])
                self.state["optimizer"].step()
                clip.model.convert_weights(self.state["model"])
            if self.config["solver"]["type"] != 'monitor':
                if (batch_index + 1) == 1 or (batch_index + 1) % 10 == 0:
                    self.state["lr_scheduler"].step(self.state["epoch"] + batch_index / len(self.state["train_loader"]))

    def on_user_valid(self):
        if self.state["epoch"] % self.config["logging"]["eval_freq"] == 0:

            # set mode
            self.state["model"].eval()
            self.state["fusion_model"].eval()

            # valid
            with torch.no_grad():
                torch.cuda.empty_cache()
                classes, num_text_aug, text_dict = TextPrompt()(self.state["val_dataset"])
                num = 0
                corr_1 = 0
                corr_5 = 0
                splits = len(classes) // 100 + 1
                # tops = [[], []]
                for iii, (image, class_id) in enumerate(track(self.state["val_loader"])):
                    image = image.view((-1, self.config["data"]["num_segments"], 3) + image.size()[-2:])
                    b, t, c, h, w = image.size()
                    image_input = image.to(self.state["device"]).view(-1, c, h, w)

                    image_features = self.state["model"].encode_image(image_input).view(b, t, -1)
                    image_features = self.state["fusion_model"](image_features)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    similarity = None
                    for _index in range(splits):
                        if _index < splits:
                            _classes = classes[_index * 100: (_index + 1) * 100, :]
                        elif _index == splits:
                            _classes = classes[_index * 100:, :]
                        text_inputs = _classes.to(self.state["device"])
                        text_features = self.state["model"].encode_text(text_inputs)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        if similarity is None:
                            similarity = (100.0 * image_features @ text_features.T)
                        else:
                            similarity = torch.cat([similarity, 100.0 * image_features @ text_features.T], 1)

                    similarity = similarity.view(b, num_text_aug, -1).softmax(dim=-1)
                    similarity = similarity.mean(dim=1, keepdim=False)
                    values_1, indices_1 = similarity.topk(1, dim=-1)
                    values_5, indices_5 = similarity.topk(5, dim=-1)
                    num += b
                    for i in range(b):
                        if list(self.state["val_dataset"].sentences.keys())[indices_1[i]] == class_id[i]:
                            Printer.print_log(
                                "{}\n{}".format(list(self.state["val_dataset"].sentences.keys())[indices_1[i]],
                                                class_id[i]))
                            corr_1 += 1
                        if class_id[i] in [list(self.state["val_dataset"].sentences.keys())[_i] for _i in
                                           indices_5[i]]:
                            corr_5 += 1

            top1 = float(corr_1) / num * 100
            top5 = float(corr_5) / num * 100

            # log
            Printer.print_panle(
                {
                    "top1": top1,
                    "top5": top5
                },
                "Epoch {}".format(self.state["epoch"])
            )

            # wandb log
            if self.config["wandb"]:
                wandb.log({"top1": top1})
                wandb.log({"top5": top5})

    def on_user_save_checkpoint(self):
        Printer.print_rule('Saving')
        _object = {
            "current": self._save_epoch(),
            # "best": self._save_best()
        }
        Printer.print_panle(_object, "saved pretrain model")
        return super().on_user_save_checkpoint()

    def _save_epoch(self):
        """
        储存每一个 epoch 的模型
        """
        torch.save(
            {
                'epoch': self.state["epoch"],
                'model_state_dict': self.state["model"].state_dict(),
                'fusion_model_state_dict': self.state["fusion_model"].state_dict(),
                'optimizer_state_dict': self.state["optimizer"].state_dict(),
            },
            os.path.join(self.state["save_dir"], "checkpoint_model_epoch_{}.pt".format(self.state["epoch"]))
        )
        return os.path.join(self.state["save_dir"], "checkpoint_model_epoch_{}.pt".format(self.state["epoch"]))

    def _save_best(self):
        """
        储存直到当前的 epoch 最好的模型
        """
        torch.save(
            {
                'epoch': self.state["epoch"],
                'model_state_dict': self.state["model"].state_dict(),
                'fusion_model_state_dict': self.state["fusion_model"].state_dict(),
                'optimizer_state_dict': self.state["optimizer"].state_dict(),
            },
            os.path.join(self.state["save_dir"], 'model_best.pt')
        )
        return os.path.join(self.state["save_dir"], 'model_best.pt')


if __name__ == "__main__":
    trainer = ActionClipTrainer()
    trainer.run_train()
