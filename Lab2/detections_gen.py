import os
import glob

def gen(path):
    imgs = glob.glob('gta/'+path+'2017/*')
    for img in imgs:
        # print('python YOLOX/tools/demo.py image -f YOLOX/exps/example/custom/yolox_s.py -c YOLOX/YOLOX_outputs/yolox_s/best_ckpt.pth --path gta/val2017/' + img.split('/')[-1] + ' --conf 0.25 --nms 0.5 --tsize 640 --device gpu --log_path '+ path)
        os.system('python YOLOX/tools/demo.py image \
                -f YOLOX/exps/example/custom/yolox_s.py \
                -c YOLOX/YOLOX_outputs/yolox_s/best_ckpt.pth \
                --path gta/'+path+'2017/' + img.split('/')[-1] + ' \
                --conf 0.25 --nms 0.5 --tsize 640  --device gpu\
                --log_path '+ path)

if __name__ == '__main__':
    gen('test')

# python YOLOX/tools/demo.py image -f YOLOX/exps/example/custom/yolox_s.py -c YOLOX/YOLOX_outputs/yolox_s/best_ckpt.pth --path gta/val2017/945.jpg --conf 0.25 --nms 0.5 --tsize 640 --log_path test
# python YOLOX/tools/demo.py image -f YOLOX/exps/example/custom/yolox_s.py -c YOLOX/YOLOX_outputs/yolox_s/best_ckpt.pth --path gta/val2017/945.jpg --conf 0.25 --nms 0.5 --tsize 640 --log_path test