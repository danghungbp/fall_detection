import json
import os
import shutil

base_dir = r'c:\Users\MSIThinGF63\Downloads\fall_detection\data\fall-detection-montreal-video-Folder- video_frames- Job 2.coco'

yaml_created = False

for split in ['train', 'valid', 'test']:
    split_dir = os.path.join(base_dir, split)
    json_path = os.path.join(split_dir, '_annotations.coco.json')
    if not os.path.exists(json_path): continue
    
    with open(json_path, encoding='utf-8') as f:
        d = json.load(f)
        
    if not yaml_created:
        yaml_path = os.path.join(base_dir, 'data.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write('names:\n')
            for cat in d['categories']:
                if cat['name'] != 'fall-detection-montreal-video':
                    f.write(f"  {cat['id']}: {cat['name']}\n")
        yaml_created = True
    
    labels_dir = os.path.join(split_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)
    images_dir = os.path.join(split_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    img_info = {img['id']: img for img in d['images']}
    
    for ann in d.get('annotations', []):
        img = img_info[ann['image_id']]
        w = img['width']
        h = img['height']
        
        bbox = ann['bbox']
        x_center = (bbox[0] + bbox[2] / 2) / w
        y_center = (bbox[1] + bbox[3] / 2) / h
        box_w = bbox[2] / w
        box_h = bbox[3] / h
        
        cat_id = ann['category_id']
        txt_path = os.path.join(labels_dir, os.path.splitext(img['file_name'])[0] + '.txt')
        with open(txt_path, 'a', encoding='utf-8') as ftxt:
            ftxt.write(f'{cat_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n')
            
    for img in d['images']:
        src = os.path.join(split_dir, img['file_name'])
        dst = os.path.join(images_dir, img['file_name'])
        if os.path.exists(src):
            shutil.move(src, dst)
            
print('COCO to YOLO conversion completed successfully!')
