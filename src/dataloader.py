import random
import torch.nn.functional as F
import cv2
import torch
import glob
import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import src.utils as utils
import yaml

from src.segment_anything.utils.transforms import ResizeLongestSide


def init_foodseg103(base_image_dir, is_val=False):
    foodseg103_data_root = os.path.join(base_image_dir, "FoodSeg103", "FoodSeg103")
    with open(os.path.join(foodseg103_data_root, "category_id.txt")) as f:
        category_lines = f.readlines()
        foodseg103_classes = [' '.join(line_data.split('\t')[1:]).strip() for line_data in category_lines]
        f.close()
    foodseg103_classes = np.array(foodseg103_classes)
    if is_val:
        foodseg103_labels = sorted(
            glob.glob(
                os.path.join(foodseg103_data_root, "Images", "ann_dir", "test", "*.png")
            )
        )
    else:
        foodseg103_labels = sorted(
            glob.glob(
                os.path.join(foodseg103_data_root, "Images", "ann_dir", "train", "*.png")
            )
        )
    foodseg103_images = [
        x.replace(".png", ".jpg").replace("ann_dir", "img_dir")
        for x in foodseg103_labels
    ]

    # if is_main_process():
    print("foodseg103[{}]: ".format('test' if is_val else 'train'), len(foodseg103_images))
    return foodseg103_classes, foodseg103_images, foodseg103_labels


def init_uecfoodpix(base_image_dir, is_val=False):
    uecfoodpix_data_root = os.path.join(base_image_dir, "UECFOODPIXCOMPLETE", "UECFOODPIXCOMPLETE", "data")
    with open(os.path.join(uecfoodpix_data_root, "category.txt")) as f:
        category_lines = f.readlines()
        uecfoodpix_classes = [' '.join(line_data.split('\t')[1:]).strip() for line_data in category_lines]
        f.close()
    uecfoodpix_classes = np.array(uecfoodpix_classes)
    if is_val:
        test_img_template = os.path.join(uecfoodpix_data_root, "UECFoodPIXCOMPLETE", "test", "img", "{id}.jpg")
        test_mask_template = os.path.join(uecfoodpix_data_root, "UECFoodPIXCOMPLETE", "test", "mask", "{id}.png")
        with open(os.path.join(uecfoodpix_data_root, "test1000.txt")) as f:
            id_lines = f.readlines()
            uecfoodpix_images = [test_img_template.format(id=id_line.split('\n')[0]) for id_line in id_lines]
            uecfoodpix_labels = [test_mask_template.format(id=id_line.split('\n')[0]) for id_line in id_lines]
            f.close()
    else:
        train_img_template = os.path.join(uecfoodpix_data_root, "UECFoodPIXCOMPLETE", "train", "img", "{id}.jpg")
        train_mask_template = os.path.join(uecfoodpix_data_root, "UECFoodPIXCOMPLETE", "train", "mask", "{id}.png")
        with open(os.path.join(uecfoodpix_data_root, "train9000.txt")) as f:
            id_lines = f.readlines()
            uecfoodpix_images = [train_img_template.format(id=id_line.split('\n')[0]) for id_line in id_lines]
            uecfoodpix_labels = [train_mask_template.format(id=id_line.split('\n')[0]) for id_line in id_lines]
            f.close()
    # if is_main_process():
    print("uecfoodpix[{}]: ".format('test' if is_val else 'train'), len(uecfoodpix_images))
    return uecfoodpix_classes, uecfoodpix_images, uecfoodpix_labels


class SemSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255  # 空标签

    def __init__(
            self,
            base_image_dir,
            image_size: int = 224,
            seg_num: int = 20,
            sem_seg_data="foodseg103||uecfoodpix",
            sem_seg_data_ratio=None,
            is_val=False,
            processor=None,
    ):
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.seg_num = seg_num
        self.sem_seg_datas = sem_seg_data.split("||")
        self.processor = processor
        self.data2list = {}
        self.data2classes = {}
        sample_nums = []
        for ds in self.sem_seg_datas:
            classes, images, labels = eval("init_{}".format(ds))(base_image_dir, is_val=is_val)
            sample_nums.append(len(images))
            self.data2list[ds] = (images, labels)
            self.data2classes[ds] = classes
        if sem_seg_data_ratio is None:
            sem_seg_data_ratio = sample_nums
        sem_seg_data_ratio = sem_seg_data_ratio[:len(self.sem_seg_datas)]
        self.sem_seg_data_ratio = utils.normalize_to_one(sem_seg_data_ratio)
        # if is_main_process() and not is_val:
        print('[SEG] datasets: {}, ratios: {}.'.format(self.sem_seg_datas, self.sem_seg_data_ratio))
        self.is_val = is_val

    def __len__(self):
        num = 0
        for dataset in self.sem_seg_datas:
            num += len(self.data2list[dataset][0])
        return num

    @staticmethod # crop
    def preprocess(x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        x = (x - SemSegDataset.pixel_mean) / SemSegDataset.pixel_std
        h, w = x.shape[-2:]
        padh = SemSegDataset.img_size - h
        padw = SemSegDataset.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x.permute(2, 1, 0)

    @staticmethod
    def enhance_mask(mask: Image.Image, increase_value: int = 50) -> Image.Image:
        """
        增强mask图像中的红色通道，使其更加显眼（仅针对红色通道中不为0的像素）。

        参数：
        mask: 输入的PIL图像，RGBA模式。
        increase_value: 红色通道需要增加的值，默认是50。

        返回：
        增强后的PIL图像。
        """
        # 转换图像为 numpy 数组
        mask_np = np.array(mask)

        # 确保图像是 RGBA 格式
        if mask_np.shape[2] != 4:
            raise ValueError("输入图像必须是RGBA格式")

        # 提取红色通道（第 0 个通道）并仅增加不为 0 的值
        red_channel = mask_np[:, :, 0]
        non_zero_mask = red_channel != 0
        red_channel[non_zero_mask] = np.clip(red_channel[non_zero_mask] + increase_value, 0, 255)

        # 将修改后的红色通道赋值回原数组
        mask_np[:, :, 0] = red_channel

        # 将 numpy 数组转换回 PIL 图像
        enhanced_mask = Image.fromarray(mask_np, 'RGBA')

        return enhanced_mask


    @staticmethod # crop
    def resize_and_crop(image: Image.Image, mask: Image.Image, target_size: int = 224) -> tuple:
        """
        将图像和mask缩放，使得最短边为target_size，然后从较长边进行中心裁剪。

        :param image: PIL格式的图像。
        :param mask: PIL格式的mask图像。
        :param target_size: 缩放后的最短边大小，默认为224像素。
        :return: 裁剪并调整大小后的图像和mask。
        """
        # 获取图像的原始大小
        width, height = image.size

        # 计算缩放比例，使得最短边为target_size
        scale_factor = target_size / min(width, height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # 缩放图像和mask
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_mask = mask.resize((new_width, new_height), Image.NEAREST)

        # 计算裁剪区域，从中心裁剪到 target_size x target_size
        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        right = left + target_size
        bottom = top + target_size

        # 裁剪图像和mask
        cropped_image = resized_image.crop((left, top, right, bottom))
        cropped_mask = resized_mask.crop((left, top, right, bottom))

        return cropped_image, cropped_mask

    def visual(self, idx):
        ds = np.random.choice(
            range(len(self.sem_seg_datas)), size=1, p=self.sem_seg_data_ratio[:len(self.sem_seg_datas)]
        )[0]
        ds = self.sem_seg_datas[ds]

        images, gt_masks = self.data2list[ds]
        image_path = images[idx]
        gt_mask_path = gt_masks[idx]
        image_path = "H:/food/UECFOODPIXCOMPLETE\data/UECFOODPIXCOMPLETE/train\img/5.jpg"
        gt_mask_path = "H:/food/UECFOODPIXCOMPLETE\data/UECFOODPIXCOMPLETE/train\mask/5.png"
        gt_mask = Image.open(gt_mask_path)
        image = Image.open(image_path)
        image, gt_mask = self.resize_and_crop(image, gt_mask, self.image_size)
        # 可视化图像和mask
        gt_mask = self.enhance_mask(gt_mask)
        self.plot_image_and_mask(image, gt_mask, idx)

    def change_format(self):
        total = 0
        for ds in self.sem_seg_datas:
            images, gt_masks = self.data2list[ds]
            for idx in tqdm(range(len(images))):
                input = self[idx]
                gt_mask = input["ground_truth_mask"]
                mask = Image.fromarray(gt_mask.numpy().astype(np.uint8))
                mask.save(gt_masks[idx])
                # mask.save(gt_masks[idx])
            # for image, mask in zip(images, gt_masks):
            #     new_image = Image.open(image)
            #     new_mask = Image.open(mask)
            #     new_image, new_mask = self.resize_and_crop(new_image, new_mask, self.image_size)
            #     new_image.save(image)
            #     new_mask.save(mask)
                total += 1
        print("Total: ", total)


    def plot_image_and_mask(self, image, mask, idx):
        """
        使用Matplotlib将图像和mask可视化。
        """
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # 显示图像
        ax[0].imshow(image)
        ax[0].set_title(f"Image Index: {idx}")
        ax[0].axis('off')

        # 将mask转换为RGB格式并显示
        mask_rgb = mask.convert("RGB")  # 如果mask不是RGB模式
        ax[1].imshow(mask_rgb)
        ax[1].set_title("Ground Truth Mask")
        ax[1].axis('off')

        plt.show()

    def __getitem__(self, idx):
        ds = np.random.choice(
            range(len(self.sem_seg_datas)), size=1, p=self.sem_seg_data_ratio[:len(self.sem_seg_datas)]
        )[0]
        ds = self.sem_seg_datas[ds]

        images, gt_masks = self.data2list[ds]
        # if not self.is_val:
        #     idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        gt_mask_path = gt_masks[idx]
        gt_mask = Image.open(gt_mask_path)
        image = Image.open(image_path)
        image, gt_mask = self.resize_and_crop(image, gt_mask, self.image_size)
        gt_mask = np.array(gt_mask)
        ori_size = tuple(image.size)[::-1]

        unique_categories_label = np.unique(gt_mask).tolist()
        # remove background and others
        if ds in ["foodseg103"]:
            if 255 in unique_categories_label:
                unique_categories_label.remove(255)
            if 0 in unique_categories_label:
                unique_categories_label.remove(0)
            if 103 in unique_categories_label:
                unique_categories_label.remove(103)
            if len(unique_categories_label) == 0:
                return self.__getitem__(0)

        elif ds in ["uecfoodpix"]:
            if 255 in unique_categories_label:
                unique_categories_label.remove(255)
            if 0 in unique_categories_label:
                unique_categories_label.remove(0)
            if 101 in unique_categories_label:
                unique_categories_label.remove(101)
            if len(unique_categories_label) == 0:
                return self.__getitem__(0)
            gt_mask = gt_mask[..., 0]
        classes = [self.data2classes[ds][class_id] for class_id in unique_categories_label]
        if len(classes) == 0:
            return self.__getitem__(0)

        gt_mask = torch.from_numpy(gt_mask).long()
        masks = []
        for class_id in unique_categories_label:
            masks.append(gt_mask == class_id)

        if len(masks) > 0:
            masks = torch.stack(masks, dim=0)
        else:
            masks = torch.zeros(0, *ori_size)
            gt_mask = torch.ones(ori_size) * self.ignore_label
        gt_mask[gt_mask != 0] = 1  # 二值标签
        # SAM 推理
        try:
            box = utils.get_bounding_box(gt_mask)
        except:
            raise ValueError("Error with the mask of the image ", image_path)
        inputs = self.processor(image, ori_size, box)
        inputs["ground_truth_mask"] = gt_mask

        return inputs


class DatasetSegmentation(Dataset):
    """
    Dataset to process the images and masks

    Arguments:
        folder_path (str): The path of the folder containing the images
        processor (obj): Samprocessor class that helps pre processing the image, and prompt

    Return:
        (dict): Dictionnary with 4 keys (image, original_size, boxes, ground_truth_mask)
            image: image pre processed to 1024x1024 size
            original_size: Original size of the image before pre processing
            boxes: bouding box after adapting the coordinates of the pre processed image
            ground_truth_mask: Ground truth mask
    """

    def __init__(self, config_file: dict, processor: Samprocessor, mode: str):
        super().__init__()
        if mode == "train":
            self.img_files = glob.glob(os.path.join(config_file["DATASET"]["TRAIN_PATH"], 'images', '*.jpg'))
            self.mask_files = []
            for img_path in self.img_files:
                self.mask_files.append(os.path.join(config_file["DATASET"]["TRAIN_PATH"], 'masks',
                                                    os.path.basename(img_path)[:-4] + ".jpg"))

        else:
            self.img_files = glob.glob(os.path.join(config_file["DATASET"]["TEST_PATH"], 'images', '*.jpg'))
            self.mask_files = []
            for img_path in self.img_files:
                self.mask_files.append(os.path.join(config_file["DATASET"]["TEST_PATH"], 'masks',
                                                    os.path.basename(img_path)[:-4] + ".jpg"))

        self.processor = processor

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index: int) -> list:
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        # get image and mask in PIL format
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        mask = mask.convert('1')
        ground_truth_mask = np.array(mask)
        original_size = tuple(image.size)[::-1]

        # get bounding box prompt
        box = utils.get_bounding_box(ground_truth_mask)
        inputs = self.processor(image, original_size, box)
        inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask)

        return inputs


def collate_fn(batch: torch.utils.data) -> list:
    """
    Used to get a list of dict as output when using a dataloader

    Arguments:
        batch: The batched dataset

    Return:
        (list): list of batched dataset so a list(dict)
    """
    return list(batch)


if __name__ == "__main__":
    # classes, images, labels = eval("init_{}".format('camo_sem_seg'))('F:\Github/autogluon\examples/automm\Conv-LoRA\datasets', is_val=False)
    # print(len(classes), len(images), len(labels))
    # image = Image.open("H:/food/foodseg103/foodseg103\Images/ann_dir/train/00000000.png")
    # image = np.array(image)
    # print(image.shape, np.unique(image))
    # image[image != 0] = 255
    # image = Image.fromarray(image)
    # image.show()
    # for channel in range(image.shape[2]):
    #     unique_values = np.unique(image[:, :, channel])
    #     print(f'{channel} 通道的唯一值：\n{unique_values}\n')