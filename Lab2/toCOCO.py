from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
import glob
import json
import os, cv2
import matplotlib.pyplot as plt


def toCOCOformat(set):
    coco = Coco()
    coco.add_category(CocoCategory(id = 3, name = 'car'))

    training_dataset = glob.glob('gta/label/'+ set +'_labels/*')

    for file in training_dataset:
        label = open(file, 'r')
        train = label.readlines()

        coco_image = CocoImage(file_name=file.split('/')[-1][:-3] + 'jpg', height=1080, width=1920)
        for i in range(len(train)):

            train[i] = train[i].strip()
            bbox = train[i].split(' ')

            x_min  = float(bbox[1])*1920 - float(bbox[3])*1920 / 2
            y_min  = float(bbox[2])*1080 - float(bbox[4])*1080 / 2
            width  = float(bbox[3])*1920
            height = float(bbox[4])*1080
            coco_image.add_annotation(
                CocoAnnotation(
                bbox=[x_min, y_min, width, height],
                category_id = 3,
                category_name='car'
                )
            )
        coco.add_image(coco_image)

    save_json(data=coco.json, save_path='instances_' + set + '.json')


def evalformat(set):
    training_dataset = glob.glob('gta/label/'+ set +'_labels/*')

    for file in training_dataset:
        label = open(file, 'r')
        train = label.readlines()

        gt = open('val/groundtruths/'+file.split('/')[-1], 'w')
        for i in range(len(train)):

            train[i] = train[i].strip()
            bbox = train[i].split(' ')

            x_min  = float(bbox[1])*1920 - float(bbox[3])*1920 / 2
            y_min  = float(bbox[2])*1080 - float(bbox[4])*1080 / 2
            width  = float(bbox[3])*1920
            height = float(bbox[4])*1080

            gt.write(f'0 {x_min} {y_min} {x_min+width} {y_min+height}\n')


        




def visualization_bbox(num_image, json_path,img_path):
    with open(json_path) as annos:
        annotation_json = json.load(annos)

    print('the annotation_json num_key is:',len(annotation_json))  
    print('the annotation_json key is:', annotation_json.keys()) 
    print('the annotation_json num_images is:', len(annotation_json['images'])) 

    for i in range(num_image):

        image_name = annotation_json['images'][i]['file_name']  
        id = annotation_json['images'][i]['id']  

        image_path = os.path.join(img_path, str(image_name).zfill(5))
        image = cv2.imread(image_path, 1) 
        num_bbox = 0 

        for i in range(len(annotation_json['annotations'][::])):
            if  annotation_json['annotations'][i-1]['image_id'] == id:
                num_bbox = num_bbox + 1
                x, y, w, h = annotation_json['annotations'][i-1]['bbox'] 
                image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)

        print('The unm_bbox of the display image is:', num_bbox)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show() 


if __name__ == "__main__":

    # toCOCOformat('train')
    # toCOCOformat('val')
    # train_json = 'instances_val.json'
    # train_path = 'dataset/val/'
    # visualization_bbox(5, train_json, train_path)

    evalformat('val')
