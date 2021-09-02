
import pytorch_lightning as pl
import torch

class BaseModel(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.learning_rate = args.learning_rate
    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        # batch
        src_ids, decoder_ids, mask, label_ids = batch
        # get loss
        loss = self(input_ids=src_ids, attention_mask=mask, decoder_input_ids=decoder_ids, labels=label_ids)
        # logs
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # batch
        src_ids, decoder_ids, mask, label_ids = batch
        # get loss
        loss = self(input_ids=src_ids, attention_mask=mask, decoder_input_ids=decoder_ids, labels=label_ids)
        self.log('validation_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log('val_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # batch
        src_ids, decoder_ids, mask, label_ids = batch
        # get loss
        loss = self(input_ids=src_ids, attention_mask=mask, decoder_input_ids=decoder_ids, labels=label_ids)
        return loss

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log('test_loss', avg_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if self.args.img_lr_factor != 1 and self.args.model=='multi_modal_bart':
            # make parameter groups
            all_para = [p for p in self.model.parameters()]
            # img_related_para = [p for p in self.model.model.encoder.img_transformer.parameters()] \
            #                   +[p for p in self.model.model.encoder.img_feature_transfers.parameters()] \
            #                   +[p for p in self.model.model.encoder.fcs.parameters()] \
            #                   +[p for p in self.model.model.encoder.final_layer_norm.parameters()] \
            #                   +[p for p in self.model.model.encoder.fgs.parameters()]

            # img_related_para = [p for p in self.model.model.encoder.img_feature_transfers.parameters()] \
            #                   +[p for p in self.model.model.encoder.fcs.parameters()] \
            #                   +[p for p in self.model.model.encoder.final_layer_norm.parameters()] \
            #                   +[p for p in self.model.model.encoder.fgs.parameters()]

            _img_related_para = []
            if self.args.cross_attn_type == 0:
                _img_related_para += [
                    self.model.model.encoder._linear_1.parameters(),
                    self.model.model.encoder._linear_2.parameters()
                ]
            elif self.args.cross_attn_type == 1:
                _img_related_para += [
                    self.model.model.encoder._linear_1.parameters(),
                    self.model.model.encoder._linear_2.parameters()
                ]
            elif self.args.cross_attn_type == 2:
                _img_related_para += [
                    self.model.model.encoder._linear_1.parameters()
                ]
            elif self.args.cross_attn_type == 3:
                _img_related_para += [
                    self.model.model.encoder._linear_1.parameters(),
                    self.model.model.encoder._linear_2.parameters(),
                    self.model.model.encoder._linear_3.parameters()
                ]
            elif self.args.cross_attn_type == 4:
                _img_related_para += [
                    self.model.model.encoder._linear_1.parameters(),
                    self.model.model.encoder._linear_2.parameters(),
                    self.model.model.encoder._linear_3.parameters(),
                    self.model.model.encoder._linear_4.parameters(),
                    self.model.model.encoder._multi_head_attn.parameters()
                ]
            elif self.args.cross_attn_type == 5:
                _img_related_para += [
                    self.model.model.encoder._linear_1.parameters(),
                    self.model.model.encoder._linear_2.parameters(),
                    self.model.model.encoder._linear_3.parameters(),
                    self.model.model.encoder._multi_head_attn.parameters()
                ]

            if self.args.use_forget_gate:
                _img_related_para.append(self.model.model.encoder.fg.parameters())

            img_related_para = []
            for params in _img_related_para:
                for param in params:
                    img_related_para.append(param)

            bart_para = []

            for p in all_para:
                flag = 0
                for q in img_related_para:
                    if p.shape == q.shape:
                        if torch.equal(p, q):
                            flag = 1
                if flag == 0:
                    bart_para.append(p)
                    continue

            optimizer = torch.optim.Adam([
                {'params': bart_para},
                {'params': img_related_para, 'lr': self.learning_rate * self.args.img_lr_factor},
            ], lr=self.learning_rate)

        elif self.args.img_lr_factor != 1 and self.args.model=='multi_modal_t5':
             # make parameter groups
            all_para = [p for p in self.model.parameters()]
            # img_related_para = [p for p in self.model.model.encoder.img_transformer.parameters()] \
            #                   +[p for p in self.model.model.encoder.img_feature_transfers.parameters()] \
            #                   +[p for p in self.model.model.encoder.fcs.parameters()] \
            #                   +[p for p in self.model.model.encoder.final_layer_norm.parameters()] \
            #                   +[p for p in self.model.model.encoder.fgs.parameters()]

            # img_related_para = [p for p in self.model.model.encoder.img_feature_transfers.parameters()] \
            #                   +[p for p in self.model.model.encoder.fcs.parameters()] \
            #                   +[p for p in self.model.model.encoder.final_layer_norm.parameters()] \
            #                   +[p for p in self.model.model.encoder.fgs.parameters()]

            _img_related_para = []
            if self.args.cross_attn_type == 0:
                _img_related_para += [
                    self.model.encoder._linear_1.parameters(),
                    self.model.encoder._linear_2.parameters()
                ]
            elif self.args.cross_attn_type == 1:
                _img_related_para += [
                    self.model.encoder._linear_1.parameters(),
                    self.model.encoder._linear_2.parameters()
                ]
            elif self.args.cross_attn_type == 2:
                _img_related_para += [
                    self.model.encoder._linear_1.parameters()
                ]
            elif self.args.cross_attn_type == 3:
                _img_related_para += [
                    self.model.encoder._linear_1.parameters(),
                    self.model.encoder._linear_2.parameters(),
                    self.model.encoder._linear_3.parameters()
                ]
            elif self.args.cross_attn_type == 4:
                _img_related_para += [
                    self.model.encoder._linear_1.parameters(),
                    self.model.encoder._linear_2.parameters(),
                    self.model.encoder._linear_3.parameters(),
                    self.model.encoder._linear_4.parameters(),
                    self.model.encoder._multi_head_attn.parameters()
                ]
            elif self.args.cross_attn_type == 5:
                _img_related_para += [
                    self.model.encoder._linear_1.parameters(),
                    self.model.encoder._linear_2.parameters(),
                    self.model.encoder._linear_3.parameters(),
                    self.model.encoder._multi_head_attn.parameters()
                ]

            if self.args.use_forget_gate:
                _img_related_para.append(self.model.encoder.fg.parameters())

            img_related_para = []
            for params in _img_related_para:
                for param in params:
                    img_related_para.append(param)

            bart_para = []

            for p in all_para:
                flag = 0
                for q in img_related_para:
                    if p.shape == q.shape:
                        if torch.equal(p, q):
                            flag = 1
                if flag == 0:
                    bart_para.append(p)
                    continue

            optimizer = torch.optim.Adam([
                {'params': bart_para},
                {'params': img_related_para, 'lr': self.learning_rate * self.args.img_lr_factor},
            ], lr=self.learning_rate)
            print('LEARNING RATE SET SUCCESSFUL')
            print('LEARNING RATE SET SUCCESSFUL')
            print('LEARNING RATE SET SUCCESSFUL')
            print('LEARNING RATE SET SUCCESSFUL')
            print('LEARNING RATE SET SUCCESSFUL')
            print('LEARNING RATE SET SUCCESSFUL')
            print('LEARNING RATE SET SUCCESSFUL')
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # return optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler_lambda1, gamma=self.args.scheduler_lambda2)
        return [optimizer], [scheduler]