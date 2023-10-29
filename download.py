import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


def download_files(url, download_folder):
    # 发送HTTP请求获取网页内容
    response = requests.get(url)

    # 使用Beautiful Soup解析HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # 遍历页面中的所有链接
    for link in soup.find_all('a', href=True):
        file_url = urljoin(url, link['href'])
        print(link)
        # 检查链接是否是文件链接
        if '.' in os.path.basename(file_url):
            # 提取文件名
            file_name = os.path.basename(file_url)

            # 构建文件的本地路径
            local_path = os.path.join(download_folder, file_name)

            # 下载文件
            with open(local_path, 'wb') as file:
                file_content = requests.get(file_url).content
                file.write(file_content)
                print(f"Downloaded: {file_name}")


# 设置要下载的网页URL和本地下载文件夹
url_to_download = 'https://huggingface.co/spaces/koajoel/PolyFormer'
download_folder = 'downloaded_files'

# 确保下载文件夹存在，如果不存在则创建
os.makedirs(download_folder, exist_ok=True)

# 调用下载函数
download_files(url_to_download, download_folder)
