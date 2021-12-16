from argparse import ArgumentParser
from ..configurations import Config
from .functions import call_vocabulary_pruning, call_transformer_pruning, call_pipeling_pruning
from .utils import create_configurations, create_model_and_tokenizer
from .utils import create_dataloader_and_adaptor
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser("TextPruner CLI tool")

    parser.add_argument("--configurations", type=str, nargs='*', help="The configurations (json files) passed to the pruner. Seperate the filenames by space. \
TextPruner uses the default configurations if omitted")
    parser.add_argument("--pruning_mode", choices=['vocabulary','transformer','pipeline'], required=True, help="One of the three pruning modes.")
    parser.add_argument("--model_class", type=str,required=True, help="The class of your model. It must be accessible from the current directory.")
    parser.add_argument("--tokenizer_class", type=str,required=True, help="The class of your tokenizer. It must be accessible from the current directory.")
    parser.add_argument("--model_path", type=str, required=True, help="The directory where the weights and the configs of the pretrained model and the tokenizer locate.")
    parser.add_argument("--vocabulary", type=str, help="A text file that is used to count tokens for vocabulay pruning.")
    parser.add_argument("--dataloader_and_adaptor",type=str, help="The script that contains the dataloader and the adaptor. \
For example: foo/bar/dataloader_script.py or foo/bar/Processing.dataloader_script (in the latter case dataloader_script is in the package Processing.")
    args = parser.parse_args()


    # initialize model and tokenizer
    model, tokenizer = create_model_and_tokenizer(model_class_name=args.model_class, 
                                                 tokenizer_class_name=args.tokenizer_class,
                                                 model_path = args.model_path)


    # initialize configurations
    configurations = create_configurations(args.configurations)


    # import functions
    dataloader, adaptor = create_dataloader_and_adaptor(args.dataloader_and_adaptor)



    if args.pruning_mode == 'vocabulary':
        call_vocabulary_pruning(configurations, model, tokenizer, args.vocabulary)
    elif args.pruning_mode == 'transformer':
         call_transformer_pruning(configurations, model, dataloader, adaptor)
    elif args.pruning_mode == 'pipeline':
         call_pipeling_pruning(configurations, model, tokenizer, args.vocabulary, dataloader, adaptor)


if __name__ == '__main__':
    main()