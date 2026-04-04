import requests
import hashlib
import time
import hmac
import json

ACCESS_ID = "r9d8vudrkyq5th47cmwe"
ACCESS_SECRET = "36c12d5e184849909b79050ae3a07e2d"
DEVICE_ID = "d71149b613b931f447fgkd"
BASE_URL = "https://openapi.tuyain.com"

def get_token():
    t = str(int(time.time() * 1000))
    msg = ACCESS_ID + t + "GET\n" + "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" + "\n\n" + "/v1.0/token?grant_type=1"
    sign = hmac.new(ACCESS_SECRET.encode('utf-8'), msg.encode('utf-8'), hashlib.sha256).hexdigest().upper()
    headers = {"client_id": ACCESS_ID, "sign": sign, "t": t, "sign_method": "HMAC-SHA256"}
    r = requests.get(BASE_URL + "/v1.0/token?grant_type=1", headers=headers)
    print(r.json())
    return r.json()["result"]["access_token"]

def control_plug(token, turn_on=True):
    t = str(int(time.time() * 1000))
    body = json.dumps({"commands": [{"code": "switch_1", "value": turn_on}]})
    body_hash = hashlib.sha256(body.encode('utf-8')).hexdigest()
    msg = ACCESS_ID + token + t + "POST\n" + body_hash + "\n\n" + "/v1.0/devices/" + DEVICE_ID + "/commands"
    sign = hmac.new(ACCESS_SECRET.encode('utf-8'), msg.encode('utf-8'), hashlib.sha256).hexdigest().upper()
    headers = {"client_id": ACCESS_ID, "access_token": token, "sign": sign, "t": t, "sign_method": "HMAC-SHA256"}
    r = requests.post(BASE_URL + f"/v1.0/devices/{DEVICE_ID}/commands", headers=headers, data=body)
    print(r.json())

token = get_token()
control_plug(token, True)
