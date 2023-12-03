import argparse

parser = argparse.ArgumentParser(description='Parameters for Training/Testing/Evaluating with the Generalizable Coherence Model')
parser.add_argument('--train_file', default='./dataset/train_pair=5_absa_Implicit_Labeled.pkl', help='Provide path to pickle file with training data. Refer to README for format.') #INSted_permuted_train_CNN.pkl
parser.add_argument('--dev_file', default='./dataset/test_pair=5_absa_Implicit_Labeled.pkl',  help='Provide path to pickle file with development data. Refer to README for format.') #INSteD_permuted_dev_CNN.pkl
parser.add_argument('--test_file', default='./dataset/test_pair=5_absa_Implicit_Labeled.pkl', help='Provide path to pickle file with test data. Refer to README for format.')
# parser.add_argument('--test_file', default='./dataset/test.pkl', help='Provide path to pickle file with test data. Refer to README for format.')
parser.add_argument('--model_size', default='base', help='Specify XLNet model size. Note that the model has only been tested with XLNet-base.')
parser.add_argument('--lr_start', type=float, default=1.3837581708618872e-05, help='Starting learning rate')

parser.add_argument('--embed_dim', type=int, default=300, help='Starting learning rate')
parser.add_argument('--hidden_dim', type=int, default=768, help='Starting learning rate')
parser.add_argument('--polarities_dim', type=int, default=3, help='Starting learning rate')

parser.add_argument('--dropout_rate', default=0.1, type=float,
                    help='specify the dropout rate for all layer, also applies to all transformer layers.')

parser.add_argument('--alpha', default=0.8167551348421718, type=float,
                    help='specify the dropout rate for all layer, also applies to all transformer layers.')

# args.alpha
parser.add_argument('--freeze_emb_layer', action='store_true',
                    help='freezes gradients updates at the embedding layer of lower transformer model.')

parser.add_argument('--sentence_pooling', type=str, default='none',
                    choices=['sum', 'mean', 'max', 'min', 'attention', 'none'],
                    help='specify the pooling strategy to use at lower transformer i.e. TF1 layer')

parser.add_argument('--lr_end', type=float, default= 1e-06, help='Final learning rate to anneal to')
parser.add_argument('--lr_anneal_epochs', type=int, default=100, help='Number of epochs/times over which to anneal the learning rate')
parser.add_argument('--eval_interval', type=int, default=50, help='Frequency of evaluation on the dev data and the LR scheduler anneal call, in number of training steps. Note that you may need to account for batch size.')
parser.add_argument('--seed', type=int, default=1234, help='Set the random seed')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--initializer', default='xavier_uniform_', type=str)
parser.add_argument('--margin', type=float, default=0.1, help='Margin for pairwise/max-margin or contrastive loss.')
parser.add_argument('--model_description', default='model_checkpoint_', help='Model description to be added to saved checkpoint filename.')
parser.add_argument('--data_type', default='multiple', help='Whether the input has a single negative or multiple negatives. Note that in case the dataset has multiple negatives, the model will only use the first negative sample. Refer to README for format.')
parser.add_argument('--num_negs', type=int, default=1, help='Specify the number of negative samples for each positive sample. This should be 1 for pairwise, 2+ for contrastive, and in the order of 25+ for the momentum model to enable hard negative mining. Use in combination with ')
parser.add_argument('--max_len', type=int, default=750, help='Maximum number of tokens (based on XLNet tokenizer) in the document. Tokens will be padded or excess will be truncated. Ensure the data is preprocessed accordingly so that truncation does not affect the task.')
parser.add_argument('--pretrained_model',  help='Path to pre-trained model. Load a pre-trained model for testing or fine-tuning.') #default='model_checkpoint__seed-100_bs-1_lr-5e-06_step-48000_acc-0.9832_type-base.pair',
#default='model_checkpoint__seed-100_bs-1_lr-5e-06_step-2570_acc-0.9018691588785047_type-base.pair',
# default='model_checkpoint__seed-100_bs-1_lr-5e-06_step-2560_acc-0.9018691588785047_type-base.pair',
# cl
# temperatureP

parser.add_argument('--temperatureP', default=0.5, type=float)
