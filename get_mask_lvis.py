from PIL import Image
import imgviz
import argparse
import os
import tqdm
from pycocotools.coco import COCO
import numpy as np
import shutil
import cv2

import logging
from lvis import LVIS, LVISResults, LVISEval
import pdb


def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)
 
def main(args):
    annotation_file = os.path.join(args.annotation_file, 'lvis_v1_{}.json'.format(args.split))
    os.makedirs(os.path.join(args.save_dir, 'Mask_Classes'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'Mask_Images_CV2'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'Mask_Images'), exist_ok=True)
    #os.makedirs(os.path.join(args.save_dir, 'Mask_Images_Stack'), exist_ok=True)
    #os.makedirs(os.path.join(args.save_dir, 'Matched_Ori_Images'), exist_ok=True)
    lvis = LVIS(annotation_file)
    catIds = lvis.get_cat_ids() # 1203 
    imgIds = lvis.get_img_ids() # 19809
    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))
    for imgId in tqdm.tqdm(imgIds, ncols=100):
        img = lvis.load_imgs([imgId])[0]
        annIds = lvis.get_ann_ids(img_ids=[img['id']], cat_ids=catIds, area_rng=None)
        anns = lvis.load_anns(annIds)
        pdb.set_trace()
        if len(annIds) > 0:
            mask = lvis.ann_to_mask(anns[0]) * anns[0]['category_id']
            pdb.set_trace()
            for i in range(len(anns) - 1):
                mask += lvis.ann_to_mask(anns[i + 1]) * anns[i + 1]['category_id']
            img_origin_path = os.path.join(args.input_dir, 'train2017', img['coco_url'][-16:])
            seg_output_path = os.path.join(args.save_dir, 'Mask_Classes', img['coco_url'][-16:].replace('.jpg', '.png'))
            #img_output_path = os.path.join(args.save_dir, 'Matched_Ori_Images', img['coco_url'][-16:])
            if os.path.exists(img_origin_path):
                original_img = np.array(Image.open(os.path.join(args.input_dir, 'train2017', img['coco_url'][-16:])).convert("RGB"))
                original_img = np.transpose(original_img, [2,0,1])
                #mask_img =  mask * original_img       
                #mask_img = np.transpose(mask_img, [1,2,0])
                #mask_img = Image.fromarray(mask_img, mode="RGB")
                #mask_img.save(os.path.join(args.save_dir, 'Mask_Images', img['coco_url'][-16:]))
                original_image = cv2.imread(os.path.join(args.input_dir, 'train2017', img['coco_url'][-16:]))
                masked = cv2.add(original_image, np.zeros(np.shape(original_image), dtype=np.uint8), mask=np.uint8(mask))
                cv2.imwrite(os.path.join(args.save_dir, 'Mask_Images', img['coco_url'][-16:]), masked)
                if args.mode == "P":
                    save_colored_mask(mask, seg_output_path)
                else:
                    cv2.imwrite(seg_output_path, mask)
               # shutil.copy(img_origin_path, img_output_path)
 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="../../data/LVIS/lvis_v1", type=str, help="img folder")
    parser.add_argument("--save_dir", default="../../data/Processed_LVIS", type=str, help="save images folder")
    parser.add_argument("--annotation_file", default="../../data/LVIS/lvis_v1/annotations", type=str, help="img folder")                      
    parser.add_argument("--split", default="train", type=str,
                        help="train2017 or val2017")
    parser.add_argument("--mode", default="L", type=str, help="save mask with rgb or black mode")
    return parser.parse_args()
 
 
if __name__ == '__main__':
    args = get_args()
    main(args)
