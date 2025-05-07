import os
import json
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.utils.data
from pathlib import Path
import skimage.io as sio
import cv2
from utils import encode_mask, decode_maskobj
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# 檢查CUDA是否可用
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# 定義資料集類別
class CellSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_ids_file=None, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        if mode == 'train':
            # 获取訓練資料夾內所有文件夾
            self.image_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            print(f"找到 {len(self.image_dirs)} 個訓練樣本目錄")
        else:
            # 測試模式
            self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.tif')]
            print(f"找到 {len(self.image_paths)} 個測試圖像")
            
            # 讀取測試圖像ID映射
            with open(image_ids_file, 'r') as f:
                self.image_id_map = json.load(f)
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.image_dirs)
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            # 訓練模式
            img_dir = os.path.join(self.root_dir, self.image_dirs[idx])
            image_path = os.path.join(img_dir, 'image.tif')
            
            # 讀取圖像
            image = sio.imread(image_path)
            
            # 確保是3通道RGB圖像
            if len(image.shape) == 2:  # 灰度圖
                image = np.stack([image, image, image], axis=2)
            elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA圖像
                image = image[:, :, :3]  # 只保留RGB通道
            elif len(image.shape) == 3 and image.shape[2] > 4:  # 多通道圖像
                # 取前三個通道
                image = image[:, :, :3]
            elif len(image.shape) == 3 and image.shape[2] < 3:  # 1或2通道
                # 擴展成3通道
                channels = [image[:, :, i] for i in range(image.shape[2])]
                while len(channels) < 3:
                    channels.append(channels[-1])
                image = np.stack(channels, axis=2)
            
            print(f"訓練圖像形狀: {image.shape}")  # 偵錯用
            
            # 初始化遮罩、框和標籤
            all_masks = []
            boxes = []
            labels = []
            
            # 遍歷所有可能的類別文件
            for class_id, class_name in enumerate(['class1.tif', 'class2.tif', 'class3.tif', 'class4.tif'], 1):
                mask_path = os.path.join(img_dir, class_name)
                if os.path.exists(mask_path):
                    mask = sio.imread(mask_path)
                    unique_vals = np.unique(mask)
                    
                    # 處理每個實例
                    for val in unique_vals:
                        if val == 0:  # 跳過背景
                            continue
                        
                        # 創建二值遮罩
                        binary_mask = (mask == val).astype(np.uint8)
                        
                        # 獲取邊界框
                        pos = np.where(binary_mask)
                        if len(pos[0]) == 0:
                            continue
                            
                        xmin, xmax = np.min(pos[1]), np.max(pos[1])
                        ymin, ymax = np.min(pos[0]), np.max(pos[0])
                        
                        # 確保框有有效大小
                        if xmax > xmin and ymax > ymin:
                            boxes.append([xmin, ymin, xmax, ymax])
                            all_masks.append(binary_mask)
                            labels.append(class_id)
            
            # 確保至少有一個實例
            if len(boxes) == 0:
                # 創建一個虛擬小物體
                h, w = image.shape[:2]
                boxes.append([0, 0, 20, 20])
                all_masks.append(np.zeros((h, w), dtype=np.uint8))
                labels.append(1)
            
            # 转換為張量
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(all_masks), dtype=torch.uint8)
            
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            
            target = {
                'boxes': boxes,
                'labels': labels,
                'masks': masks,
                'image_id': image_id,
                'area': area,
                'iscrowd': iscrowd
            }
            
            # 將圖像轉換為PyTorch張量
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
            
            return image, target
        
        else:
            # 測試模式
            image_path = self.image_paths[idx]
            image_filename = os.path.basename(image_path)
            
            # 從映射查找image_id
            image_id = None
            image_width = None
            image_height = None
            for item in self.image_id_map:
                if item['file_name'] == image_filename:
                    image_id = item['id']
                    image_width = item['width']
                    image_height = item['height']
                    break
            
            # 讀取圖像
            image = sio.imread(image_path)
            
            # 確保是3通道RGB圖像
            if len(image.shape) == 2:  # 灰度圖
                image = np.stack([image, image, image], axis=2)
            elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA圖像
                image = image[:, :, :3]  # 只保留RGB通道
            elif len(image.shape) == 3 and image.shape[2] > 4:  # 多通道圖像
                # 取前三個通道
                image = image[:, :, :3]
            elif len(image.shape) == 3 and image.shape[2] < 3:  # 1或2通道
                # 擴展成3通道
                channels = [image[:, :, i] for i in range(image.shape[2])]
                while len(channels) < 3:
                    channels.append(channels[-1])
                image = np.stack(channels, axis=2)
            
            # 將圖像轉換為PyTorch張量
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
            
            return image, {'image_id': image_id, 'file_name': image_filename, 'width': image_width, 'height': image_height}

# 明確定義collate函數
def collate_fn(batch):
    return tuple(zip(*batch))

# 建立Mask R-CNN模型
def get_model_instance_segmentation(num_classes):
    # 使用預訓練的模型
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights)
    
    # 替換分類頭
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # 替換遮罩頭
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model

# 訓練一個epoch
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    
    total_loss = 0
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 偵錯用：檢查輸入圖像形狀
        for img in images:
            print(f"訓練批次圖像形狀: {img.shape}")
        
        try:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # 反向傳播與優化
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
            
            if i % print_freq == 0:
                print(f"Epoch: {epoch}, Batch: {i}/{len(data_loader)}, Loss: {losses.item():.4f}")
        except Exception as e:
            print(f"訓練批次發生錯誤: {e}")
            # 繼續下一個批次
            continue
    
    avg_loss = total_loss / max(1, len(data_loader))
    print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
    return avg_loss

# 生成預測結果
@torch.no_grad()
def generate_predictions(model, data_loader, device):
    model.eval()
    results = []
    
    for images, info in data_loader:
        images = list(img.to(device) for img in images)
        
        try:
            outputs = model(images)
            
            for idx, output in enumerate(outputs):
                image_id = info[idx]['image_id']
                image_height = info[idx]['height'] 
                image_width = info[idx]['width']
                
                # 收集預測結果
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                pred_masks = output['masks'].cpu().numpy()
                
                # 只保留高分的預測
                keep = pred_scores > 0.5
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]
                pred_masks = pred_masks[keep]
                
                # 處理每個預測
                for box, score, label, mask in zip(pred_boxes, pred_scores, pred_labels, pred_masks):
                    # 二值化遮罩
                    binary_mask = (mask[0] > 0.5).astype(np.uint8)
                    
                    # 確保遮罩大小正確
                    if binary_mask.shape[0] != image_height or binary_mask.shape[1] != image_width:
                        binary_mask = cv2.resize(binary_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    
                    # 如果遮罩為空，跳過
                    if np.sum(binary_mask) == 0:
                        continue
                    
                    # 編碼為RLE格式
                    rle = encode_mask(binary_mask)
                    
                    # 創建結果項
                    result = {
                        'image_id': int(image_id),
                        'category_id': int(label),
                        'bbox': box.tolist(),
                        'score': float(score),
                        'segmentation': {
                            'size': [int(image_height), int(image_width)],
                            'counts': rle['counts']
                        }
                    }
                    
                    results.append(result)
        except Exception as e:
            print(f"預測過程中出錯: {e}")
            # 繼續下一個批次
            continue
    
    return results

def main():
    # 設定路徑
    train_dir = 'hw3-data-release/train'
    test_dir = 'hw3-data-release/test_release'
    image_ids_file = 'hw3-data-release/test_image_name_to_ids.json'
    
    # 數據集和數據加載器
    print("Creating datasets...")
    train_dataset = CellSegmentationDataset(
        root_dir=str(train_dir), 
        mode='train'
    )
    
    test_dataset = CellSegmentationDataset(
        root_dir=str(test_dir),
        image_ids_file=str(image_ids_file),
        mode='test'
    )
    
    # 檢查數據集是否成功創建
    if len(train_dataset) == 0:
        raise ValueError("訓練數據集為空，請檢查路徑和數據結構")
    
    # 創建數據加載器 - 使用明確定義的collate_fn
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=0,  # 單處理模式
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # 創建模型
    print("Creating model...")
    num_classes = 5  # 背景 + 4個類別
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    
    # 創建優化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # 訓練模型
    num_epochs = 10
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # 訓練
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        # 更新學習率
        lr_scheduler.step()
        
        # 每個epoch保存一次模型
        torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
    
    # 在測試集上生成預測
    print("Generating predictions...")
    results = generate_predictions(model, test_loader, device)
    
    # 保存結果為JSON
    with open('test-results.json', 'w') as f:
        json.dump(results, f)
    
    print(f"結果已保存到 test-results.json，共 {len(results)} 個預測")

if __name__ == "__main__":
    main()

