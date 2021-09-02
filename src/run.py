import argparse
from data_preprocess.data_builder import SummaryDataModule
from models.bart import BartOrigin
from models.t5 import T5Origin, T5MultiModal
from models.multi_modal_model import BartMultiModal
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin


if __name__ == '__main__':
    # for training
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', default='text_only_bart', type=str, help='We have for models to choose, text_only_bart, multi_modal_bart,  text_only_t5 and multi_modal_t5')
    parser.add_argument('-checkpoint', default='./lightning_logs/version_7/checkpoints/epoch=29-step=4499.ckpt', type=str, help='The checkpoint path')
    parser.add_argument('-train_src_path', default='./dataset/sum_train/tran.tok.txt', type=str, help='training input data path (dialogue)')
    parser.add_argument('-train_tgt_path', default='./dataset/sum_train/desc.tok.txt', type=str, help='training output data path (summary)')
    parser.add_argument('-val_src_path', default='./dataset/sum_cv/tran.tok.txt', type=str, help='validatioin input data path (dialogue)')
    parser.add_argument('-val_tgt_path', default='./dataset/sum_cv/desc.tok.txt', type=str, help='validatioin output data path (summary)')
    parser.add_argument('-test_src_path', default='./dataset/sum_devtest/tran.tok.txt', type=str, help='testing input data path (dialogue)')
    parser.add_argument('-test_tgt_path', default='./dataset/sum_devtest/desc.tok.txt', type=str, help='testing output data path (summary)')
    parser.add_argument('-image_feature_path', default='./dataset/video_action_features/', type=str, help='video features path')
    parser.add_argument('-val_save_file', default='./evaluation/temp_valid_file', type=str, help='the validation results for each epoch')
    parser.add_argument('-test_save_file', default='./evaluation/results/test_summaries.txt', type=str, help='the generated summary for testing data')
    parser.add_argument('-log_name', default='multi_modal_bart', type=str, help='lightning log path')
    parser.add_argument('-gpus', default='0,2,3,4', type=str, help='choose gpus to run the code, you can choose multipple gpus')
    parser.add_argument('-batch_size', type=int, default=4, help='batch size for each gpu')
    parser.add_argument('-max_input_len', type=int, default=512, help='the maximun length for input dialogue')
    parser.add_argument('-max_output_len', type=int, default=64, help='the maximun length for output summary')
    parser.add_argument('-max_img_len', type=int, default=256, help='the maximun length for video features')
    parser.add_argument('-n_beams', type=int, default=4, help='the number of beams using for generation')
    parser.add_argument('-no_repeat_ngram_size', type=int, default=3, help='the size of no repeat ngrams during generation')
    parser.add_argument('-learning_rate', default=3e-5, type=float, help='learning rate')
    parser.add_argument('-scheduler_lambda1', default=20, type=int, help='change the learning each lambda1 epoch')
    parser.add_argument('-scheduler_lambda2', default=0.95, type=float, help='the learning rate will times lambda2 for each change')
    parser.add_argument('-num_epochs', type=int, default=100, help='maximun number of training epoches')
    parser.add_argument('-grad_accumulate', type=int, default=10, help='gradient accumulation for this number iterations')
    parser.add_argument('-random_seed', type=int, default=0, help='global random seed')
    parser.add_argument('-do_train', type=str, default='True', help='set True to training, set False to not training')
    parser.add_argument('-do_test', type=str, default='True', help='set True to testing, set False to not testing')
    parser.add_argument('-limit_val_batches', default=1.0, type=float, help='do validation for each epoch')
    parser.add_argument('-val_check_interval', type=float, default=1, help='do validation for each epoch')
    parser.add_argument('-img_lr_factor', type=float, default=1, help='the learning rate for visual guidance part will times this number')

    # About cross-modal attention and fusion
    parser.add_argument('-use_img_trans', action='store_true', help='whether or not to use VTF')
    parser.add_argument('-use_forget_gate', action='store_true', help='whether or not to use forget gate')
    parser.add_argument('-fusion_layer', type=int, default=5, help='number of fusion layers') # 5 is the last layer
    '''
    Textual features: T in (S_t, D_t)
    Visual features: V in (S_v, D_v)

    cross_attn_type == 0
    => V' = linear_1(V) in (S_v, D_t)
    => A = Dot_Prod_Attn(T, V') in (S_t, S_v)
    => A = softmax(A)
    => T_out = linear_2(concat(T, AV)) in (S_t, D_t)
    => T_out = T + T_out (Residual Connection)

    cross_attn_type == 1
    => V' = linear_1(V) in (S_v, D_t)
    => A = Dot_Prod_Attn(T, V') in (S_t, S_v)
    => A = softmax(A)
    => T_out = linear_2(concat(T, AV')) in (S_t, D_t), only this step is different from 0
    => T_out = T + T_out (Residual Connection)

    cross_attn_type == 2
    => V' = linear_1(V) in (S_v, D_t)
    => A = Dot_Prod_Attn(T, V') in (S_t, S_v)
    => A = softmax(A)
    => T_out = AV'
    => T_out = T + T_out (Residual Connection)

    cross_attn_type == 3
    => T' = linear_1(T) in (S_t, D_a), D_a << D_t
    => V' = linear_2(V) in (S_v, D_a), D_a << D_v
    => A = Dot_Prod_Attn(T', V') in (S_t, S_v)
    => A = softmax(A)
    => T_out = linear_3(concat(T, AV)) in (S_t, D_t)
    => T_out = T + T_out (Residual Connection)

    cross_attn_type == 4
    => K_a = linear_1(V) in (S_v, D_a)
    => V_a = linear_2(V) in (S_v, D_a)
    => Q_a = linear_3(T) in (S_t, D_a)
    => T_out = MultiHeadAttn(Q_a, K_a, V_a) in (S_t, D_a)
    => T_out = linear_4(concat(T, T_out)) in (S_t, D_t)
    => T_out = T + T_out (Residual Connection)

    cross_attn_type == 5 (Only valid when D_a == D_t)
    => K_a = linear_1(V) in (S_v, D_a)
    => V_a = linear_2(V) in (S_v, D_a)
    => Q_a = linear_3(T) in (S_t, D_a)
    => T_out = MultiHeadAttn(Q_a, K_a, V_a) in (S_t, D_a)
    => T_out = T + T_out (Residual Connection)
    '''
    parser.add_argument('-cross_attn_type', type=int, default=0)
    parser.add_argument('-dim_common', type=int, default=256)
    parser.add_argument('-n_attn_heads', type=int, default=1) # only needed when cross_attn_type is 4 or 5

    # Add to decoding
    parser.add_argument('-fusion_in_decoding', action='store_true')
    parser.add_argument('-vision_use_noise', action='store_true')

    args = parser.parse_args()

    # random seed
    seed_everything(args.random_seed)

    # set logger
    logger = pl_loggers.TensorBoardLogger(f'./lightning_logs/{args.log_name}')

    # save checkpoint
    checkpoint_callback = ModelCheckpoint(monitor='validation_Rouge2_one_epoch',
                                          save_last=True,
                                          save_top_k=2,
                                          mode='max',)

    # make trainer
    if args.checkpoint == 'None':
        args.checkpoint = None
    trainer = Trainer(deterministic=True,
                      num_sanity_val_steps=10,
                      resume_from_checkpoint=args.checkpoint,
                      logger=logger,
                      gpus=args.gpus,
                      distributed_backend='ddp',
                      plugins=DDPPlugin(find_unused_parameters=False),
                      gradient_clip_val=1.0,
                      max_epochs=args.num_epochs,
                      limit_val_batches=args.limit_val_batches,
                      val_check_interval=args.val_check_interval,
                      accumulate_grad_batches=args.grad_accumulate,
                      fast_dev_run=False,
                      callbacks=[checkpoint_callback])

    # make dataloader & model
    summary_data = SummaryDataModule(args)
    if args.model == 'text_only_bart':
        model = BartOrigin(args)
    elif args.model == 'multi_modal_bart':
        model = BartMultiModal(args)
    elif args.model == 'text_only_t5':
        model = T5Origin(args)
    elif args.model == 'multi_modal_t5':
        model = T5MultiModal(args)
    else:
        raise ValueError("Invalid model")

    # Fit the instantiated model to the data
    if args.do_train == 'True':
        trainer.fit(model, train_dataloader=summary_data.train_loader, val_dataloaders=summary_data.val_loader)
    if args.do_test == 'True':
        model = model.load_from_checkpoint(args.checkpoint, args=args)
        trainer.test(model=model, test_dataloaders=summary_data.test_loader)