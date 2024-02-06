'''Script that loades data with given urls and preprocesses it for future use'''
import os
import sys
import requests
import shutil
from zipfile import ZipFile

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import get_root_dir

DATA_PATH = get_root_dir('data')



def load_data(urls:list):
    # Check if the 'data' folder exists, create it if not
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    flag = True
    for url in urls:
        # Download the zip file
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # Save the zip file
            zip_file_path = os.path.join(DATA_PATH, 'train_data.zip')
            with open(zip_file_path, 'wb') as file:
                shutil.copyfileobj(response.raw, file)

            # Extract the contents of the zip file
            with ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_PATH)

            # Remove the temporary zip file
            os.remove(zip_file_path)
            if flag:
                os.rename(os.path.join(DATA_PATH,'final_project_train_dataset'),os.path.join(DATA_PATH,'raw'))
            else:
                source = os.path.join(DATA_PATH,'final_project_test_dataset/test.csv')
                dest = os.path.join(DATA_PATH,'raw')
                shutil.copy2(source,dest)
                shutil.rmtree(os.path.join(DATA_PATH,'final_project_test_dataset'))
            flag = False
        else:
            print(f'Failed to download data. Status code: {response.status_code}')





if __name__ == "__main__":
    urls = ['https://static.cdn.epam.com/uploads/583f9e4a37492715074c531dbd5abad2/ds/final_project_train_dataset.zip',
    'https://static.cdn.epam.com/uploads/583f9e4a37492715074c531dbd5abad2/ds/final_project_test_dataset.zip',]

    load_data(urls)


