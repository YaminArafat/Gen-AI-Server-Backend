import requests
import time

BASE_URL = "http://192.168.0.68:5000/"
endpoint = f"{BASE_URL}/create/config/from/text"

payload = {
    "inputText": "make a simplified config"
}
testInput = ["make a simplified config", "ben10 animated minimalistic config", "saturn planet gif config with digital clock", "tom and jerry animation config with exercise related widget in all edges", "ghost gif background config with digital minimal config"]

task_ids = []

for i, prompt in enumerate(testInput):
    print(f"Sending request for {i+1}...")
    response = requests.post(endpoint, json={"inputText": prompt})
    print(response.json())
    if response.status_code == 200:
        data = response.json()
        task_id = data['task_id']
        task_ids.append(task_id)
        print(f"Task submitted , task ID of {i+1}: {task_id}")
        print(data)
    else:
        print(f"Request {i+1} failed : {response.text}")
print("All tasks submitted.")
completed = set()
while len(completed) < len(task_ids):
    for task_id in task_ids:
        if task_id in completed:
            continue
        status_result = requests.get(f"{BASE_URL}/status/{task_id}")
        if status_result.status_code == 200:
            status_info = status_result.json()
            print(f"Task {task_id} status: {status_info["status"]}, progress: {status_info["progress"]}")
            if status_info["status"] in ["Completed", "Failed"]:
                completed.add(task_id)
        else:
            print(f"Error while check task status for task_id {task_id}")
        time.sleep(3)
print("All tasks Completed.")



