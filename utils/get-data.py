import os
import shutil
from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np

def download_dataset_from_roboflow(api_key: str, workspace: str, project: str, version: int, model: str, location: str):
    """
    Downloads a dataset from Roboflow.

    Args:
        api_key (str): The Roboflow API key.
        workspace (str): The Roboflow workspace name.
        project (str): The Roboflow project name.
        version (int): The version of the dataset to download.
        location (str): The local directory to save the dataset.
    """
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download(model, location)

    return dataset

def create_dataset_instance(dataset, split: str):
    """
    Creates a DetectionDataset instance from a YOLO formatted dataset.

    Args:
        dataset: The dataset object returned by Roboflow.

    Returns:
        A DetectionDataset instance.
    """
    ds = sv.DetectionDataset.from_yolo(
        images_directory_path=f"{dataset.location}/{split}/images",
        annotations_directory_path=f"{dataset.location}/{split}/labels",
        data_yaml_path=f"{dataset.location}/data.yaml"
    )
    return ds

def merge_and_save_datasets(dataset_list, images_directory_path, annotations_directory_path, data_yaml_path):
    """
    Merges a list of DetectionDataset objects into a single DetectionDataset object and saves it in YOLO format.

    Args:
        dataset_list (list): A list of DetectionDataset objects to merge.
        images_directory_path (str): The directory to save the images.
        annotations_directory_path (str): The directory to save the annotations.
        data_yaml_path (str): The path to save the data.yaml file.
    """
    # Merge the datasets
    merged_dataset = sv.DetectionDataset.merge(dataset_list)

    # Save the merged dataset in YOLO format
    merged_dataset.as_yolo(
        images_directory_path=images_directory_path,
        annotations_directory_path=annotations_directory_path,
        data_yaml_path=data_yaml_path
    )

if __name__ == "__main__":
    api_key = "OyAGy3t2A8sGTkQ2HLSj"
    workspace = {
        'traffic_light' : ["traffic-light-collection", "traffic-signs-data-analytics"],
        'vehicle' : ['transportation-collection-1', 'transportation-collection-2'],
        'crosswalk' : ['crosswalk-collection']
    }
    project = {
        'traffic_light' : ["traffic-light-collection", "data-analytics-ncbsw"],
        'vehicle' : ['traffic-vehicle-collection', 'traffic-transportation-collection'],
        'crosswalk' : ['crosswalk-collection']
    }
    version = {
        'traffic_light' : 2,
        'vehicle' : [2, 1],
        'crosswalk' : 1
    }
    
    # Download the datasets
    datasets = {
        'traffic_light': [],
        'vehicle': [],
        'crosswalk': []
    }
    
    # Create base directory for downloads
    base_dir = "data"
    model = "yolov12"
    os.makedirs(base_dir, exist_ok=True)
    
    # Download traffic light datasets
    for ws, proj in zip(workspace['traffic_light'], project['traffic_light']):
        dataset = download_dataset_from_roboflow(
            api_key=api_key,
            workspace=ws,
            project=proj,
            version=version['traffic_light'],
            model=model,
            location=f"{base_dir}/traffic_light_{proj}"
        )
        datasets['traffic_light'].append(dataset)
    
    # Download vehicle datasets
    for ws, proj, ver in zip(workspace['vehicle'], project['vehicle'], version['vehicle']):
        dataset = download_dataset_from_roboflow(
            api_key=api_key,
            workspace=ws,
            project=proj,
            version=ver,
            model=model,
            location=f"{base_dir}/vehicle_{proj}"
        )
        datasets['vehicle'].append(dataset)
    

    # Download crosswalk dataset
    dataset = download_dataset_from_roboflow(
        api_key=api_key,
        workspace=workspace['crosswalk'][0],
        project=project['crosswalk'][0],
        version=version['crosswalk'],
        model=model,
        location=f"{base_dir}/crosswalk"
    )
    datasets['crosswalk'].append(dataset)
    
    # Create merged dataset instances for each split
    splits = ['train', 'valid', 'test']
    merged_base_dir = "merged_data"
    os.makedirs(merged_base_dir, exist_ok=True)
    
    for split in splits:
        # Create split directory
        split_dir = os.path.join(merged_base_dir, split)
        os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "labels"), exist_ok=True)
        
        # Get dataset instances for current split
        all_datasets = []
        for category in datasets:
            for dataset in datasets[category]:
                try:
                    ds = create_dataset_instance(dataset, split)
                    all_datasets.append(ds)
                except Exception as e:
                    print(f"Warning: Could not create dataset instance for {category} - {split}: {str(e)}")
        
        # Merge and save datasets for current split
        if all_datasets:
            merge_and_save_datasets(
                dataset_list=all_datasets,
                images_directory_path=os.path.join(split_dir, "images"),
                annotations_directory_path=os.path.join(split_dir, "labels"),
                data_yaml_path=os.path.join(merged_base_dir, "data.yaml")
            )
            print(f"Successfully merged {split} datasets")

