import os
import json
import cv2

# convert predicted images from yolov8s to json
# change bbox format of yolov8 [x_center, y_center, w, h] to [x_top_left, y_top_left, w, h]

def get_image_Id(img_name):
    img_name = img_name.split('.txt')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx)+str(sceneIndx)+str(frameIndx))
    return imageId

preds_txt_path = 'output\infer_results_yolov8s_fisheye1k_public_test\labels'
preds_img_path = 'output\infer_results_yolov8s_fisheye1k_public_test\images'

txt_files = os.listdir(preds_txt_path)
# print(txt_file[0:5])
# print(txt_file[0])
# print(get_image_Id(txt_file[0]))
# print(len(txt_file))


# # Ảnh k có output camera24_A_72

# img_file = os.listdir('output\infer_results_yolov8s_fisheye1keval\images')
# print(len(img_file))

# for img in img_file:
#     # print(img)
#     img2txt = img.split('.png')[0] + ".txt"
#     # print(img2txt)
#     if img2txt not in txt_file:
#        print(img)

final_results = []

for txt_file in txt_files:
    # get Id
    img_path = preds_img_path + "\\" + txt_file.split('.txt')[0] + ".png"
    # print(img_path)
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    # print(type(height))
    # print(type(width))
    # print(height)
    # print(width)
    # break
    img_id = get_image_Id(img_name=txt_file)
    # print(img_id)

    # read pred json
    txt_path = preds_txt_path + '\\' + txt_file
    #   print(txt_path)
  
    with open(txt_path, 'r') as file:
        for line in file:
            # print(line)
            c, x, y, w, h, sc = line.strip().split()
            x1 = (float(x)-float(w)/2) * width
            y1 = (float(y)-float(h)/2) * height
            # print(values)
            tmp_dict = {
                "image_id": img_id,
                "category_id": int(c),
                "bbox": [round(x1,6), round(y1,6), round(float(w) * width,6), round(float(h) * height,6)],
                "score": float(sc)
                }
            if float(sc) >= 0.4:
                final_results.append(tmp_dict)
            # break

    # break

# print(final_results)

# Đường dẫn của file JSON
file_path = 'output\infer_results_yolov8s_fisheye1k_public_test\infer_results_yolov8s_thrshld40_fisheye1keval.json'

# Ghi mảng vào file JSON
with open(file_path, 'w') as json_file:
    json.dump(final_results, json_file)

print("DONE!")