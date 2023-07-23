from os import replace
import torch
import torchvision
import shutil
from framework_activity_recognition.io import make_model_dir, make_logger, make_logging_dir, make_benchmark_dir
from torch.utils.data import DataLoader
from framework_activity_recognition.wrapper import QuantizationAwareTrainingWrapper, BenchmarkWrapper
from torch.utils.tensorboard import SummaryWriter
from framework_activity_recognition.processing import get_entity_by_module_path
from collections import OrderedDict


def train(config_file):
    """
    This method constructs training wrapper based on given configuration file
    Arguments:
        config_file: configuration file to construct the wrapper
    """
    if "student" in config_file:
        student_config = config_file["student"]
    else:
        student_config = config_file["architecture"]

    if "teacher" in config_file:
        teacher_config = config_file["teacher"]
    #teacher_config = config_file["teacher"]
    train_config = config_file["train"]
    data_config = config_file["data"]
    pretraining_config = config_file["pretraining"]

    model_dir = make_model_dir(config_file)

    logging_dir = make_logging_dir(config_file)

    logger = make_logger(logging_dir / "log.txt")
    
    #load training and test set using function in datautils.py (it can be different based on config file)
    train_set, test_set = get_entity_by_module_path(data_config["util_location"])(config_file)

    logger.info("Data:" + config_file["data"]["name"])

    annotation_converter = None

    #get annotation converter from training set if there is one (there should be one according to the Beispielcode)
    if hasattr(train_set, "annotation_converter"):
        annotation_converter = train_set.annotation_converter

    #preparing to load pretrained model on student architecture
    if "model_num_classes" in pretraining_config:
        train_config["num_classes"] = pretraining_config["model_num_classes"]
    else:
        train_config["num_classes"] = pretraining_config["student_model_num_classes"]

    student_network = get_entity_by_module_path(student_config["location"])(config_file, student_config)

    #load pretraining model on student network, teacher should use already trained model on a respective task
    if pretraining_config["use"]:

        #load the pretrained model on student network
        student_network = load_state_dictionary(student_network, config_file, student_config)

        #deactivate autograd of all layers on student network
        if pretraining_config["fine_tune_only_last_layer"]:
            for param in filter(lambda p: p.requires_grad, student_network.parameters()):
                param.requires_grad = False


    train_config["num_classes"] = train_set.nClasses

    num_classes = train_set.nClasses

    #replace out size of last layer with the number of class in training set
    student_network = replace_last_layer(student_network, config_file, student_config)

    logger.info("Loaded student architecture:" + student_config["location"].split('.')[-1])

    if data_config["sampler"]["use"]:
        sampler = get_entity_by_module_path(data_config["sampler"]["location"])(train_set)
        train_loader = DataLoader(train_set, sampler=sampler, batch_size=train_config["batch_size"], shuffle=False, num_workers=train_config["num_workers"])
    else:
        train_loader = DataLoader(train_set, batch_size=train_config["batch_size"], shuffle=True, num_workers=train_config["num_workers"])

    test_loader = DataLoader(test_set, batch_size=train_config["batch_size"], shuffle=False, num_workers=train_config["num_workers"])
    #determine loss criterion for student loss
    criterion = getattr(torch.nn, train_config["criterion"]["name"])()
    
    #create tensorboard writer
    writer = SummaryWriter(logging_dir)

    current_epoch = 0

    
    wrapper_input = {
        'student_network': student_network,
        'num_classes': num_classes,
        'config_file': config_file,
        'writer': writer,
        'logger': logger,
        'annotation_converter': annotation_converter,
        'criterion': criterion,
        'current_epoch': current_epoch,
        'max_epochs': train_config["epoch"],
        'test_rate': train_config["test_rate"],    
    }

    #copy config file to the directory with the best and latest model
    shutil.copyfile(config_file["config"]["path"], logging_dir / "config.yaml")


    if train_config["modality"] == "train_st" or train_config["modality"] == "train_st_quant" or train_config["modality"] == "train_qd":
        #preparing to load pretrained model on teacher architecture
        train_config["num_classes"] = pretraining_config["teacher_model_num_classes"]

        teacher_network = get_entity_by_module_path(teacher_config["location"])(config_file, teacher_config)

        train_config["num_classes"] = num_classes    

        teacher_network = replace_last_layer(teacher_network, config_file, teacher_config)

        #load teacher model
        teacher_state_dict = torch.load(teacher_config["checkpoint"])
        if "model_state_dict" in teacher_state_dict:
            teacher_network.load_state_dict(teacher_state_dict["model_state_dict"])
        elif "student_model_state_dict" in teacher_state_dict:
            teacher_network.load_state_dict(teacher_state_dict["student_model_state_dict"])

        wrapper_input["teacher_network"] = teacher_network
        wrapper_input["temperature"] = train_config["temperature"]
        wrapper_input["teacher_weight"] = train_config["teacher_weight"]

        logger.info("Loaded teacher architecture:" + teacher_config["location"].split('.')[-1])

    quantizationFramework = False

    if train_config["modality"] == "train_quant" or train_config["modality"] == "train_st_quant":
        quantizationFramework = True

        student_network = torch.quantization.fuse_modules(student_network, train_config["quantization"]["fuse_module"])

        student_network.qconfig = torch.quantization.get_default_qat_qconfig(train_config["quantization"]["backend"])

        torch.quantization.prepare_qat(student_network, inplace=True)

        wrapper_input["student_network"] = student_network

        wrapper_input["quantization_framework"] = quantizationFramework

        wrapper_input["freeze_bn"] = train_config["quantization"]["freeze_bn"]

        wrapper_input["freeze_observer"] = train_config["quantization"]["freeze_observer"]

    #determine optimizer for student
    optimizer = getattr(torch.optim, train_config["optimizer"]["name"])\
        (filter(lambda p: p.requires_grad, student_network.parameters()), **train_config["optimizer"]["parameter"])

    wrapper_input["optimizer"] = optimizer

    scheduler = None

    #determine scheduler for student
    if train_config["scheduler"]["use"]:
        scheduler = getattr(torch.optim.lr_scheduler, train_config["scheduler"]["name"])(optimizer, **train_config["scheduler"]["parameter"])
        wrapper_input["scheduler"] = scheduler

    wrapper = QuantizationAwareTrainingWrapper(**wrapper_input)

    wrapper.train(train_loader, test_loader)

def test_benchmark(config_file):
    """
    This method constructs test wrapper based on given configuration file
    Arguments:
        config_file: configuration file to construct the wrapper
    """
    architecture_config = config_file["architecture"]
    data_config = config_file["data"]
    train_config = config_file["train"]

    benchmark_dir = make_benchmark_dir(config_file)
    logger = make_logger(benchmark_dir / "log.txt")

    test_set = get_entity_by_module_path(data_config["util_location"])(config_file)

    train_config["num_classes"] = test_set.nClasses
    
    architecture = get_entity_by_module_path(architecture_config["location"])(config_file, architecture_config)

    num_classes = test_set.nClasses

    quantization_framework = False

    architecture_state_dict = torch.load(architecture_config["model"] + "/best_model.pth")
    if "model_state_dict" in architecture_state_dict:
        architecture.load_state_dict(architecture_state_dict["model_state_dict"])
    elif "student_model_state_dict" in architecture_state_dict:
        architecture.load_state_dict(architecture_state_dict["student_model_state_dict"])
    else:
        quantization_framework = True
        architecture = torch.quantization.fuse_modules(architecture, architecture_state_dict["fuse_module"])

        architecture.qconfig = torch.quantization.get_default_qat_qconfig(train_config["quantization"]["backend"])

        torch.quantization.prepare_qat(architecture, inplace=True)

        architecture = torch.quantization.convert(architecture.to('cpu').eval(), inplace=False)

        architecture.load_state_dict(architecture_state_dict["int_student_model_state_dict"])

    test_loader = DataLoader(test_set, batch_size=train_config["batch_size"], shuffle=False, num_workers=train_config["num_workers"], pin_memory=train_config["pin_memory"])

    wrapper = BenchmarkWrapper(architecture, num_classes, config_file, logger, architecture_state_dict["annotation"], quantization_framework)

    wrapper.benchmark(test_loader)
    


def load_state_dictionary(architecture, config_file, architecture_config):
    """
    Loads pretrained model based on architecture name

    architecture: nn.Module
        architecture to be loaded with its pretrained model
    config_file: File
        configuration file to the respective training
    
    Returns
    -------
    nn.Module: loaded_architecture
        architecture that is loaded with its pretrained model
    """
    checkpoint = torch.load(config_file["pretraining"]["path"])
    #print(len(checkpoint['state_dict']))
    loaded_architecture = architecture
    #print(sum(p.numel() for p in loaded_architecture.parameters()))
    if architecture_config["location"].split(".")[-1] == "I3D" or architecture_config["location"].split(".")[-1] == "I3DLogit":
        loaded_architecture.load_state_dict(checkpoint)
        #architecture.load_state_dict(checkpoint)
    else:
        dictWithoutModule = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            dictWithoutModule[k[7:]] = v
        loaded_architecture.load_state_dict(dictWithoutModule)
        #architecture.load_state_dict(checkpoint['state_dict'])
    return loaded_architecture
    #return architecture

def replace_last_layer(architecture, config_file, architecture_config):
    """
    Loads pretrained model based on architecture name

    architecture: nn.Module
        architecture whose last layer is to be replaced
    config_file: File
        configuration file to the respective training
    
    Returns
    -------
    nn.Module: replaced_architecture
        architecture whose last layer replaced with the new one
    """
    replaced_architecture = architecture
    
    if architecture_config["location"].split(".")[-1] == "I3D" or architecture_config["location"].split(".")[-1] == "I3DLogit":
        setattr(replaced_architecture, config_file["pretraining"]["last_layer_variable"], \
                    get_entity_by_module_path(config_file["pretraining"]["last_layer_class"])(out_channels=config_file["train"]["num_classes"], **config_file["pretraining"]["last_layer_parameter"]))
    else:
        in_features = architecture.classifier[-1].in_features
        out_features = config_file["train"]["num_classes"]
        replaced_architecture.classifier[-1] = torch.nn.Linear(in_features, out_features)

    return replaced_architecture