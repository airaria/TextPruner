import logging
from ..configurations import Config
import importlib
import importlib.machinery
import sys,os
sys.path.append(os.getcwd())
logger = logging.getLogger(__name__)


def import_factory(model_class_name: str):
    module_name, class_name = model_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name,package=None)
    try:
        cls = getattr(module, class_name)   
    except AttributeError:
        logger.info(f"Cannot get {class_name} for {module}, return None")
        cls = None
    return cls


def get_class(class_name: str):
    if len(class_name.split('.'))==1:
        class_name = 'transformers.' + class_name
    return import_factory(class_name)


def create_from_class(model_class_name: str, model_path: str):
    model_class = get_class(model_class_name)
    model = model_class.from_pretrained(model_path)
    return model


def read_file_line_by_line(texts_file: str):
    '''
    Read the file line by line. if the file contains only digits, treat the digits as the token_ids.

    Args:
        text_file : a text file that contains texts or ids.

    Returns:
        (List[str], False) if the file contains normal texts; (List[int], True) if the file contains only digits.
    '''
    lines = []
    is_token_ids = False
    with open(texts_file,'r') as f:
        for line in f:
            sline = line.strip()
            if len(sline)>0:
                lines.append(sline)
    try:
        token_ids = [int(token) for token in lines]
    except ValueError:
        is_token_ids = False
        return lines, is_token_ids
    logger.info("All contexts are digits. Treat them as the token ids.")
    is_token_ids = True
    return token_ids, is_token_ids


def create_configurations(configurations_list):
    configurations_dict = {"GeneralConfig": None, "VocabularyPruningConfig": None, "TransformerPruningConfig": None}

    if configurations_list is not None:
        for configuration_file in configurations_list:
            configuration = Config.from_json(configuration_file)
            configurations_dict[configuration.config_class] = configuration
    return configurations_dict


def create_model_and_tokenizer(model_class_name: str, tokenizer_class_name: str, model_path: str):
    model = create_from_class(model_class_name, model_path)
    tokenizer = create_from_class(tokenizer_class_name, model_path)
    return model, tokenizer


def create_dataloader_and_adaptor(dataloader_and_adaptor_script: str):
    if dataloader_and_adaptor_script is None:
        return None, None
    if os.path.sep in dataloader_and_adaptor_script:
        dirname = os.path.dirname(dataloader_and_adaptor_script)
        filename = os.path.basename(dataloader_and_adaptor_script)
        if filename.endswith('.py'):
            filename = filename[:-3]
        sys.path.insert(0, os.path.abspath(dirname))
        dataloader_name = filename + '.dataloader'
        adaptor_name = filename + '.adaptor'
    else:
        dataloader_name = dataloader_and_adaptor_script + '.dataloader'
        adaptor_name = dataloader_and_adaptor_script + '.adaptor'

    dataloader = import_factory(dataloader_name)
    adaptor = import_factory(adaptor_name)

    return dataloader, adaptor