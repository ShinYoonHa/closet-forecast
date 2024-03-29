# Color 분석을 위한 모듈
pip install colorthief
pip install webcolors

# Detection을 위한 Tensorflow 모듈 설치
import glob
#rgb값을 name으로 출력하기 위함
import webcolors

import cv2
import tensorflow as tf
import tensorflow_hub as hub
import os

#이미지 색상추출
from colorthief import ColorThief
import colorsys

# 이미지를 위한 모듈
import matplotlib
import matplotlib.pyplot as plt #차트 그리기
matplotlib.use("Agg")
import tempfile # 임시 파일/디렉터리 생성
import pandas as pd
# For drawing onto the image. 배열
import numpy as np

# 파이썬 이미지 처리 pkg
from PIL import Image # 이미지 생성, 불러오기 등 제공
from PIL import ImageColor # color table 제공
from PIL import ImageDraw # 이미지에 텍스트 쓰기
from PIL import ImageFont # 이미지 내 폰트 설정
from PIL import ImageOps # 이미지 변형

# 이미지를 url로부터 다운 받은 후 resizing.
def download_and_resize_image(url, display=False):
    _, filename = tempfile.mkstemp(suffix=".jpg")
    pil_image = Image.open(url)  # 이미지 오픈하여 pil_image에 저장 후
    pil_image_rgb = pil_image.convert("RGB")  # RBG로 변환
    pil_image_rgb.save(filename, format="JPEG", quality=90)  # JPEG로 저장
    #if display:
    #    display_image(pil_image)
    return filename

### 이미지 내 bounding box 생성
def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, font, thickness=3, display_str_list=()):
    # 이미지에 bounding box 좌표 설정
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size  # 이미지 원본 사이즈
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)

    # bounding box 테두리 그리기
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
              width=thickness, fill=color)
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]  # box string 높이가 이미지 높이 초과 시, 하단에 폰트 삽입
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)  # 각 strings에 0.05 마진

    # 각 bounding box의 top 좌표가 전체 string height보다 크다면,
    # string을 bounding box위로 보내고, 아니면 bounding box 아래에 삽입
    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height

    # list를 역방향으로 바꾸고, 아래에서 위로 출력
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str, fill="black", font=font)
        text_bottom -= text_height - 2 * margin


### 박스 그리기. 박스의 좌표를 찾고, 그 좌표 위에서 선언된 함수를 이용
def draw_boxes(path, image, boxes, class_names, scores, max_boxes=8, min_score=0.1):
    # 색상 및 폰트 설정
    print(os.path.basename(path))
    fileName=os.path.basename(path)
    colors = list(ImageColor.colormap.values())
    font = ImageFont.load_default()

    # 실제로 바운딩 박스 그리기 적용
    cnt=0
    topScore=0
    bottomScore=0
    hasSet = False
    dominant_class=[]
    class_color_set=[]
    d = {"image": [path], "top": None, "topColor": None, "bottom": None, "bottomColor": None, "one":None, "oneColor": None}
    #myDict = {"image": None, "top": None, "topColor": None, "bottom": None, "bottomColor": None, "one":None, "oneColor":None} #비어있는 딕셔너리 생성
    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score: #사물을 구문하는 최소 확률보다 큰지 확인
            ymin, xmin, ymax, xmax = tuple(boxes[i])  # 박스 좌표값
            display_str = "{}: {}%".format(class_names[i].decode("ascii"), int(100 * scores[i]))
            if hasSet == False: #찾은 한벌옷 없을 때
              if class_names[i].decode("ascii") == "Dress" or class_names[i].decode("ascii") == "Suit": #한벌옷일때
                  hasSet = True #한벌옷 찾았다
                  d = extraction(path, image, class_names, scores, fileName,i, xmin, xmax, ymin, ymax, font, display_str, colors, cnt, d, category='one')
            
              elif class_names[i].decode("ascii") == "Coat" or class_names[i].decode("ascii") == "Shirt" or class_names[
                  i].decode("ascii") == "Jacket" or class_names[i].decode("ascii") == "Clothing": #상의일 때
                if scores[i] >= topScore : #이미 찾은 다른 상의보다 확률이 높으면 추출
                    topScore = scores[i]
                    d = extraction(path, image, class_names, scores, fileName,i, xmin, xmax, ymin, ymax, font, display_str, colors, cnt, d, category='top')
                      
              elif class_names[i].decode("ascii") == "Trousers" or class_names[i].decode("ascii") == "Jeans" or class_names[
                  i].decode("ascii") == "Shorts"or class_names[i].decode("ascii") == "Skirt"or class_names[i].decode("ascii") == "Miniskirt":
                if scores[i] >= bottomScore : #다른 하의보다 확률이 높으면 추출
                    bottomScore = scores[i]
                    d = extraction(path, image, class_names, scores, fileName,i, xmin, xmax, ymin, ymax, font, display_str, colors, cnt, d, category='bottom')

    df = pd.DataFrame(d)
    df.to_csv('detection_dominant_color_category_test.csv', index=False, mode='a', encoding='utf-8-sig', header=False)
    return image

def extraction(path, image, class_names, scores, fileName,i, xmin, xmax, ymin, ymax, font, display_str, colors, cnt, d, category):
                color = colors[hash(class_names[i]) % len(colors)]
                image_pil = Image.fromarray(np.uint8(image)).convert("RGB")

                im_width, im_height = image_pil.size  # 이미지 원본 사이즈
                (left, right, top, bottom) = (
                xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)  # x,y의 min, max는 원본사이즈 대비 비율임.

                area = (left, top, right, bottom)
                op_img = Image.open(path)
                crop_img = op_img.crop(area)
                # print("주요 색깔 전")
                crop_img.save("/savedCrop/"+"_"+str(cnt)+"_"+fileName)  # 괄호 속 이름으로 crop된 이미지를 저장하겠다!
                plt.figure(figsize=(8, 8))
                # plt.imshow(crop_img)  # 인자로numpy 값 필요. draw_boxes 함수 입력인자 중 image가 numpy값임
                # plt.show()

                ct_img = ColorThief("/savedCrop/"+"_"+str(cnt)+"_"+fileName)  # 원래 파일 주소가 인자로 넘어가야함
                dominant_color = ct_img.get_color(quality=1)

                cnt=cnt+1

                draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font,
                                           display_str_list=[display_str])
                np.copyto(image, np.array(image_pil))
                if not os.path.exists('detection_dominant_color_category_test.csv'):
                      df = pd.DataFrame(d)
                      df.to_csv('detection_dominant_color_category_test.csv', index=False, mode='w', encoding='utf-8-sig')
                
                if category == 'top':
                      d.update(top = class_names[i].decode("ascii"))
                      d.update(topColor = get_color_name(dominant_color))
                elif category == 'bottom':
                      d.update(bottom = class_names[i].decode("ascii"))
                      d.update(bottomColor = get_color_name(dominant_color))
                else :
                      d.update(one = class_names[i].decode("ascii"))
                      d.update(oneColor = get_color_name(dominant_color))

                return d

# 최종 이미지 출력
def display_image(image):
    fig = plt.figure(figsize=(10, 10))
    plt.grid(False)
colorName = {'#900000': 'red', '#933131': 'red', '#FC5353': 'red', '#FF7E7E': 'red',
             
      '#FFCF6A': 'orange', '#FFAC00': 'orange', '#EF9F18': 'orange', '#F1B91B': 'orange',

      '#F5F17B': 'yellow', '#F7EF0C': 'yellow', '#CDC705': 'yellow', '#DCDB24': 'yellow', '#DFCA92' : 'yellow',

      '#A5C935' : 'green', '#95C935' : 'green', '#7AC935' : 'green', '#66C935' : 'green',
      '#4C8446' : 'green', '#558446' : 'green', '#6B8446' : 'green', '#768446' : 'green',
      '#9CC688' : 'green', '#347017' : 'green', '#177046' : 'green',

      '#0DF0E0' : 'blue', '#0DB3F0' : 'blue', '#0D7FF0' : 'blue', '#0D63F0' : 'blue',
      '#0D47F0' : 'blue', '#0D21F0' : 'blue', '#190DF0' : 'blue', '#3B42AA' : 'blue',
      '#3B5AAA' : 'blue', '#3B73AA' : 'blue', '#3B95AA' : 'blue', '#3BA9AA' : 'blue', '#5A7891' : 'blue',
      '#87CECA' : 'blue', '#64A19E' : 'blue', '#2C4A7B' : 'blue', '#505579' : 'blue', '#6873BC' : 'blue',

      '#4F11E2' : 'purple', '#5F11E2' : 'purple', '#8211E2' : 'purple', '#9611E2' : 'purple',
      '#A911E2' : 'purple', '#8C40A9' : 'purple', '#9C40A9' : 'purple', '#8640A9' : 'purple',
      '#9B76AD' : 'purple', '#561B73' : 'purple', '#5C4169' : 'purple', '#58208C' : 'purple',
      '#9870D0' : 'purple', '#63229D' : 'purple', '#601FDD' : 'purple', '#454383' : 'purple',

      '#BB1FDD' : 'violet', '#CC1FDD' : 'violet', '#DD1FDD' : 'violet', '#DD1FC5' : 'violet',
      '#D055B3' : 'violet', '#D055CC' : 'violet', '#ED7FE9' : 'violet', '#DC7FED' : 'violet',
      '#EABBEB' : 'violet', '#FF9BBD' : 'violet', '#C60FA1' : 'violet', '#C474BD' : 'violet',
      '#D71D89' : 'violet', '#F56FB0' : 'violet', '#AC8691' : 'violet',

      '#E45129' : 'brown', '#A2391D' : 'brown', '#864A3A' : 'brown', '#AC5B37' : 'brown',
      '#884A1C' : 'brown', '#C07F2F' : 'brown', '#A25114' : 'brown', '#A97147' : 'brown',
      '#7B674C' : 'brown', '#645045' : 'brown',

      '#F8E4CA' : 'beige', '#F8E9CA' : 'beige', '#EFE1D7' : 'beige', '#DECEC3' : 'beige',
      '#A49789' : 'beige', '#F2E2CF' : 'beige', '#EDDAB3' : 'beige', '#E4C7A9' : 'beige',
      '#E8D295' : 'beige', '#CAB097' : 'beige',

      '#191717' : 'black', '#1B0C0C' : 'black', '#1D0303' : 'black', '#031D11' : 'black',
      '#1E2421' : 'black', '#241E21' : 'black', '#320B2C' : 'black', '#32250B' : 'black',
      '#0B2C32' : 'black', '#2F2B27' : 'black', '#16281F' : 'black', '#33333F' : 'black',
      '#3F333C' : 'black', '#37311A' : 'black', '#392532' : 'black',

      '#787373' : 'grey', '#485859' : 'grey', '#514F4F' : 'grey', '#686868' : 'grey',
      '#2F3130' : 'grey', '#3F3F38' : 'grey', '#41413D' : 'grey', '#2D3E38' : 'grey',

      '#FFF9F2' : 'white', '#F2FFF5' : 'white', '#F0F1F0' : 'white', '#EBEBE0' : 'white', '#CFCFD0' : 'white',
      '#E1E0EB' : 'white', '#E2E2E1' : 'white', '#EEEEE7' : 'white', '#F7F7F3' : 'white', '#BABABA' : 'white'
             }

def closest_color(requested_color):
    min_colors = {}
    for key, name in colorName.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_color_name(requested_color):
    try:
        closest_name = webcolors.rgb_to_name(requested_color)
    except ValueError:
        closest_name = closest_color(requested_color)
    return closest_name

module_handle1 = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" # FasterRCNN+InceptionResNet V2: 정확도 높음

detector_faster_Rcnn = hub.load(module_handle1).signatures['default'] #detector에 사용할 모듈 저장

def load_img(path):
  img = tf.io.read_file(path)   # tf로 파일 읽기
  img = tf.image.decode_jpeg(img, channels=3)   # 파일로부터 jpeg read
  return img


### 사용모듈과 img path를 통해 detection 실행
def run_detector(detector, path):
  img = load_img(path) #image 불러오기
  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...] #tf.float32로 img 변형
  result = detector(converted_img) #변형된 타입의 이미지를 detection
  result = {key:value.numpy() for key,value in result.items()}

  # 이미지 내 박스로 entity 및 socres를 추가하여 출력
  image_with_boxes = draw_boxes(path, img.numpy(), result["detection_boxes"], result["detection_class_entities"], result["detection_scores"])

folder_dir="/content/test"
for images in glob.iglob(f'{folder_dir}/*'):

    # check if the image ends with png
    suffixes = ('.png', '.jpg')
    if (images.endswith(suffixes)):
        run_detector(detector_faster_Rcnn, images)