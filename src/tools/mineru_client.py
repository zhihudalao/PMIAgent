from ast import main
from src.config.loader import get_str_env


mineru_api_base = get_str_env("MINERU_API_BASE")
mineru_api_key = get_str_env("MINERU_API_KEY")


import requests

def url_api():

    token = mineru_api_key
    url = f"{mineru_api_base}/api/v4/extract/task/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"url":"https://cdn-mineru.openxlab.org.cn/demo/example.pdf", "data_id": "abcd"}
        ],
        "model_version": "vlm"
    }
    try:
        response = requests.post(url,headers=header,json=data)
        if response.status_code == 200:
            result = response.json()
            print('response success. result:{}'.format(result))
            if result["code"] == 0:
                batch_id = result["data"]["batch_id"]
                print('batch_id:{}'.format(batch_id))
            else:
                print('submit task failed,reason:{}'.format(result.msg))
        else:
            print('response not success. status:{} ,result:{}'.format(response.status_code, response))
    except Exception as err:
        print(err)

def task_api(batch_id: str):
    token = mineru_api_key
    url = f"{mineru_api_base}/api/v4/extract-results/batch/{batch_id}"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    try:
        response = requests.get(url,headers=header)
        if response.status_code == 200:
            result = response.json()
            print('response success. result:{}'.format(result))
            if result["code"] == 0:
                task_id = result["data"]["task_id"]
                print('task_id:{}'.format(task_id))
            else:
                print('get task failed,reason:{}'.format(result.msg))
        else:
            print('response not success. status:{} ,result:{}'.format(response.status_code, response))
    except Exception as err:
        print(err)

if __name__ == "__main__":
    #url_api()
    task_api(batch_id="e222d241-9fc3-43d4-bfa7-60144c307f6c")
