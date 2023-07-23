import torch
import torchvision
import torchvision.transforms as transforms
import os
from framework_activity_recognition.videotransform import RandomSelect, RandomCrop, RandomHorizontalFlip,\
    normalizeColorInputZeroCenterUnitRange, CenterCrop, ToTensor
from framework_activity_recognition.parser import parseHMDBSplitFiles, parseDriveNActSplitFiles, parseDriveNActTestSplitFiles
from framework_activity_recognition.dataset import VideoFileBasedDataset, DriveNActDataset

def prepare_drivenact(config_file):
    """
    Construct Drive&Act train and validation Dataset instance based on the given configuration file
    Arguments:
        config_file: configuration file to construct Dataset instance
    """
    split_nr = config_file["data"]["split_nr"]
    folder_splits = config_file["data"]["folder_splits"]
    data_split = config_file["data"]["data_path"]
    views = config_file["data"]["views"]
    new_suffix = config_file["data"]["new_suffix"]
    n_frame = config_file["data"]["n_frame"]
    frame_size = config_file["data"]["frame_size"]
    train_df, test_df = parseDriveNActSplitFiles(split_nr, folder_splits, data_split, views, new_suffix)

    transform_training = torchvision.transforms.Compose([
        RandomSelect(n=n_frame),
        RandomCrop(height=frame_size, width=frame_size),
        RandomHorizontalFlip(),
        normalizeColorInputZeroCenterUnitRange(),
        ToTensor()
    ])

    transform_test = torchvision.transforms.Compose([
        RandomSelect(n=n_frame),
        CenterCrop(height=frame_size, width=frame_size),
        normalizeColorInputZeroCenterUnitRange(),
        ToTensor()
    ])

    annotations_train = train_df['activity'].tolist()
    annotations_test = test_df['activity'].tolist()

    dataset_train = DriveNActDataset(train_df, annotations_train, None, transform_training)

    annotation_converter = dataset_train.__get_annotation_converter__()

    dataset_test = DriveNActDataset(test_df,annotations_test,annotation_converter,transform_test)

    return dataset_train, dataset_test

def prepare_drivenact_test(config_file):
    """
    Construct Drive&Act test Dataset instance based on the given configuration file
    Arguments:
        config_file: configuration file to construct Dataset instance
    """
    split_nr = config_file["data"]["split_nr"]
    folder_splits = config_file["data"]["folder_splits"]
    data_split = config_file["data"]["data_path"]
    views = config_file["data"]["views"]
    new_suffix = config_file["data"]["new_suffix"]
    n_frame = config_file["data"]["n_frame"]
    frame_size = config_file["data"]["frame_size"]

    test_df = parseDriveNActTestSplitFiles(split_nr, folder_splits, data_split, views, new_suffix)

    transform_test = torchvision.transforms.Compose([
        RandomSelect(n=n_frame),
        CenterCrop(height=frame_size, width=frame_size),
        normalizeColorInputZeroCenterUnitRange(),
        ToTensor()
    ])

    annotations_test = test_df['activity'].tolist()

    dataset_test = DriveNActDataset(test_df,annotations_test,None,transform_test)

    return dataset_test





