import glob, os

def evalformat():
    training_dataset = glob.glob('../gta/test_labels/*.txt')

    os.makedirs('../gta/SE', exist_ok = True)
    os.makedirs('../gta/SE/groundtruths', exist_ok = True)
    os.makedirs('../gta/SE/detections', exist_ok = True)

    for file in training_dataset:
        label = open(file, 'r')
        train = label.readlines()

        gt = open('../gta/SE/groundtruths/'+file.split('/')[-1], 'w')
        for i in range(len(train)):

            train[i] = train[i].strip()
            bbox = train[i].split(' ')

            x_min  = float(bbox[1])*1920 - float(bbox[3])*1920 / 2
            y_min  = float(bbox[2])*1080 - float(bbox[4])*1080 / 2
            width  = float(bbox[3])*1920
            height = float(bbox[4])*1080

            gt.write(f'0 {x_min} {y_min} {x_min+width} {y_min+height}\n')

def inference():
    os.system("python YOLOX/tools/demo.py image -f YOLOX/exps/example/custom/yolox_s.py -c best_ckpt_SE_e600.pth --path ../gta/test --conf 0.35 --nms 0.5 --tsize 640 --log_path '../gta/SE/'")

if __name__ == "__main__":

    evalformat()
    inference()
