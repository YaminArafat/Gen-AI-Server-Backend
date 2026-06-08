import requests

BASE_URL = "http://192.168.0.68:5000/"
endpoint = f"{BASE_URL}/create/config/from/text"

payload = {
    "inputText": "I want a animated look with digital time, a subtle date box at six o'clock, health widgets around all the edges and elegant cyan circle widget color with moonphase. Mouth out save report want box recognize strong. Work individual he then candidate easy here specific."
}

response = requests.post(endpoint, json=payload)

if response.status_code == 200:
    data = response.json()
    print("Output received: ")
    print(data)
else:
    print("Error: ", response.status_code, response.text)