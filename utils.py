import os
import requests
import urllib.request
from io import BytesIO
from urllib.parse import urlparse

def buff_png (image):
    buff = BytesIO()
    image.save(buff, format = 'PNG')
    buff.seek(0)
    return buff

def upload_image (url, image):
    response = requests.put(url, data = buff_png(image), headers = { 'Content-Type': 'image/png' })
    response.raise_for_status()

def extract_origin_pathname (url):
    parsed_url = urlparse(url)
    origin_pathname = parsed_url.scheme + '://' + parsed_url.netloc + parsed_url.path
    return origin_pathname

def rounded_size (width, height):
    rounded_width = (width // 8) * 8
    rounded_height = (height // 8) * 8

    if width % 8 >= 4:
        rounded_width += 8
    if height % 8 >= 4:
        rounded_height += 8

    return int(rounded_width), int(rounded_height)

def sc (self, clip_input, images):
    return images, [False for i in images]

def download_url (url, folder_path):
    # 获取文件名
    file_name = url.split('/')[-1]

    # 构造保存路径
    save_path = os.path.join(folder_path, 'image.png')

    # 下载文件
    urllib.request.urlretrieve(url, save_path)

    print(f"文件已保存到: {save_path}")
