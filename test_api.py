import requests

# Replace with your actual server IP/port
BASE_URL = "http://192.168.0.68:5000"
endpoint = f"{BASE_URL}/api/upload_audio"

file_path = ".\datasets\liridon_eminem-lose-yourself.mp3"

with open(file_path, 'rb') as f:
    files = {
        'audio': ("liridon_eminem-lose-yourself.mp3", f, 'audio/mpeg')
    }
    response = requests.post(endpoint, files=files)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())

# payload = {
#     "inputText": "I want a animated look with digital time, a subtle date box at six o'clock, health widget around all the edges and elegant cyan circle widget color with moonphase. Mouth out save report want box recognize strong. Work individual he then candidate easy here specific."
# }

# response = requests.post(endpoint, json=payload)

# if response.status_code == 200:
#     data = response.json()
#     print("JSON text returned:")
#     print(data)
# else:
#     print("Error:", response.status_code, response.text)
