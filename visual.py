import yaml

from src.dataloader import SemSegDataset
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b

with open("./config.yaml", "r") as ymlfile:
    config_file = yaml.load(ymlfile, Loader=yaml.Loader)
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
baseline = sam
processor = Samprocessor(baseline)
# dataset = DatasetSegmentation(config_file, processor, mode="test")
# print(dataset.visualize(5))
data = SemSegDataset(base_image_dir='H:/food', processor=processor)
data.change_format()
