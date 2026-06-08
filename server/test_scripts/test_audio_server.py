import requests

BASE_URL = "http://192.168.0.68:5000/"
endpoint = f"{BASE_URL}/create/config/from/audio"


file_path = "../datasets/liridon_eminem-lose-yourself.mp3"

with open(file_path, 'rb') as f:
    files = {
        'audio': ("liridon_eminem-lose-yourself.mp3", f, 'audio/mpeg')  # use 'audio/mpeg' for .mp3
    }
    response = requests.post(endpoint, files=files)

print("Status Code:", response.status_code)
print("Response JSON:", response)