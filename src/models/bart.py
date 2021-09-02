from models.base_model import BaseModel
from transformers import BartTokenizer
from transformers import BartForConditionalGeneration
from datasets import load_metric

class BartOrigin(BaseModel):

    def __init__(self,args):
        self.args = args
        super(BartOrigin, self).__init__(args)
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.rouge = load_metric('rouge', experiment_id=self.args.log_name)

    def forward(self, input_ids, attention_mask, decoder_input_ids, labels):
        loss = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)[0]
        return loss

    def validation_step(self, batch, batch_idx):
        # batch
        src_ids, decoder_ids, mask, label_ids = batch
        # get summary
        summary_ids = self.model.generate(input_ids=src_ids,
                                            attention_mask=mask,
                                            num_beams=self.args.n_beams,
                                            max_length=self.args.max_output_len,
                                            early_stopping=True,
                                            no_repeat_ngram_size=self.args.no_repeat_ngram_size)
        return [summary_ids, label_ids]

    def validation_epoch_end(self, outputs):
        summary = []
        reference = []
        for item in outputs:
            summary_id = item[0]
            label_id = item[1]
            one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
            one_reference = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in label_id]
            summary += one_summary
            reference += one_reference
        avg_rouge1, avg_rouge2, avg_rougeL = self.calrouge(summary, reference, self.rouge)
        self.log('validation_Rouge1_one_epoch', avg_rouge1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_Rouge2_one_epoch', avg_rouge2, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_RougeL_one_epoch', avg_rougeL, on_epoch=True, prog_bar=True, sync_dist=True)
        self.save_txt(self.args.val_save_file+'_reference', reference)
        self.save_txt(self.args.val_save_file+'_summary', summary)

    def test_step(self, batch, batch_idx):
        # batch
        src_ids, decoder_ids, mask, label_ids = batch
        # get summary
        summary_ids = self.model.generate(input_ids=src_ids,
                                            attention_mask=mask,
                                            num_beams=self.args.n_beams,
                                            max_length=self.args.max_output_len,
                                            early_stopping=True,
                                            no_repeat_ngram_size=self.args.no_repeat_ngram_size)
        return [summary_ids, label_ids]

    def test_epoch_end(self, outputs):
        rouge = load_metric('rouge', experiment_id=self.args.log_name)
        summary = []
        reference = []
        for item in outputs:
            summary_id = item[0]
            label_id = item[1]
            one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
            one_reference = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in label_id]
            summary += one_summary
            reference += one_reference
        avg_rouge1, avg_rouge2, avg_rougeL = self.calrouge(summary, reference, rouge)
        self.log('test_Rouge1_one_epoch', avg_rouge1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_Rouge2_one_epoch', avg_rouge2, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_RougeL_one_epoch', avg_rougeL, on_epoch=True, prog_bar=True, sync_dist=True)
        self.save_txt(self.args.test_save_file, summary)

    def calrouge(self, summary, reference, rouge):
        rouge.add_batch(predictions=summary, references=reference)
        final_results = rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])
        R1_F1 = final_results["rouge1"].mid.fmeasure * 100
        R2_F1 = final_results["rouge2"].mid.fmeasure * 100
        RL_F1 = final_results["rougeL"].mid.fmeasure * 100
        return R1_F1, R2_F1, RL_F1

    def save_txt(self, file_name, list_data):
        file = open(file_name, 'w')
        list_data = [item+'\n' for item in list_data]
        file.writelines(list_data)
        file.close()
