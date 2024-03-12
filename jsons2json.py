import os
import json

# convert predicted images from mmdetection to json
# change bbox format of mmdetection 
#    [x_top_left, y_top_left, x_bottom_right, y_bottom_right] 
# to [x_top_left, y_top_left, w, h]
def get_image_Id(img_name):
  img_name = img_name.split('.png')[0]
  sceneList = ['M', 'A', 'E', 'N']
  cameraIndx = int(img_name.split('_')[0].split('camera')[1])
  sceneIndx = sceneList.index(img_name.split('_')[1])
  frameIndx = int(img_name.split('_')[2])
  imageId = int(str(cameraIndx)+str(sceneIndx)+str(frameIndx))
  return imageId

preds_json_path = 'output\infer_results_new-w-faster-rcnn-r101-fpn-2x-coco\preds'
vis_img_path = 'output\infer_results_new-w-faster-rcnn-r101-fpn-2x-coco\\vis'

Imgs_name = os.listdir(vis_img_path)
# print(Imgs_name[0:10])
# print(get_image_Id(Imgs_name[0]))

preds_json = os.listdir(preds_json_path)

final_results = []

for Img_name in Imgs_name:
    # get image ID
    img_id = get_image_Id(img_name=Img_name)
    # print(img_id)

    # read pred json
    json_path = preds_json_path + '\\' + Img_name.split('.')[0] + '.json'
    # print(json_path)
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    labels = data['labels']
    scores = data['scores']
    bboxes = data['bboxes']

    for lb, sc, bb in zip(labels, scores, bboxes):
        x_top_left, y_top_left, x_bot_right, y_bot_right = bb
        tmp_dict = {
            "image_id": img_id,
            "category_id": lb,
            "bbox": [x_top_left, y_top_left, x_bot_right-x_top_left, y_bot_right-y_top_left],
            "score": sc
        }
        final_results.append(tmp_dict)
    #     break
    # break
      


# print(final_results)
       
# Đường dẫn của file JSON
file_path = 'output\infer_results_new-w-faster-rcnn-r101-fpn-2x-coco\\infer_results_new_w_faster-rcnn_r101_fpn_fisheye1keval.json'

# Ghi mảng vào file JSON
with open(file_path, 'w') as json_file:
    json.dump(final_results, json_file)

print("Done!")