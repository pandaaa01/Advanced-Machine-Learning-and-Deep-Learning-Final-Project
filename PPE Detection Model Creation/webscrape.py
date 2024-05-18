import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urlparse

def download_image(url, folder):
    try:
        if url.startswith('data:image'):
            print("Skipping image with data URI.")
            return
        
        if not url.startswith(('http://', 'https://')):
            print(f"Invalid URL format: {url}")
            return
        
        r = requests.get(url, stream=True)
        r.raise_for_status()
        filename = os.path.join(folder, os.path.basename(urlparse(url).path))
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Error downloading image: {e}")


def scrape_images(keyword, num_images):
    base_url = f"https://www.google.com/search?q={keyword}&tbm=isch"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(base_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    img_tags = soup.find_all('img', limit=num_images)

    folder = "construction_worker_images"
    if not os.path.exists(folder):
        os.makedirs(folder)

    for img_tag in img_tags:
        img_url = img_tag.get('src')
        if img_url:
            download_image(img_url, folder)

if __name__ == "__main__":
    keyword = "construction worker with no helmet"
    num_images = 10
    scrape_images(keyword, num_images)
