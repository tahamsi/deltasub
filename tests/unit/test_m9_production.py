import json
from pathlib import Path

from PIL import Image
import pytest

from deltasub.experiment.campaign import CARS, load_campaign, resolve, status
from deltasub.experiment.training import ProductionManifestDataset


def _record(path, *, split, labelled, target, status="known"):
    return {"image_path":path,"train_or_test_split":split,"labelled_or_unlabelled":labelled,
            "original_class_id":target,"sample_id":f"{split}-{target}","known_or_novel":status}


def test_training_dataset_does_not_expose_unlabelled_target(tmp_path):
    Image.new("RGB",(256,256)).save(tmp_path/"image.jpg")
    records=[_record("image.jpg",split="train",labelled="unlabelled",target=91,status="novel")]
    dataset=ProductionManifestDataset(records,tmp_path,split="train",seed=0,train=True)
    assert dataset[0]["target"] == -1
    assert dataset[0]["labelled"] is False


def test_evaluation_labels_are_separate_from_training_boundary(tmp_path):
    Image.new("RGB",(256,256)).save(tmp_path/"image.jpg")
    records=[_record("image.jpg",split="test",labelled="unlabelled",target=7)]
    item=ProductionManifestDataset(records,tmp_path,split="test",seed=0,train=False)[0]
    assert "views" not in item and item["target"] == 7


def test_campaign_resolution_and_cars_status(tmp_path):
    config=load_campaign("configs/publication/core.yaml")
    run=resolve(config,"cub","baseline",0)
    assert run["dataset"]["classes"] == 200
    report=status("configs/publication/core.yaml")
    assert report["cars"] == CARS


def test_full_ablation_resolution():
    config=load_campaign("configs/publication/full.yaml")
    assert resolve(config,"aircraft","deltasub",2,"k8")["deltasub"]["fixed_k"] == 8
    with pytest.raises(ValueError): resolve(config,"aircraft","baseline",2,"k8")
