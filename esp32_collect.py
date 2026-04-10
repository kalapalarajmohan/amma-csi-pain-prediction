import socket
import datetime
DATA= "/home/rajmohan/amma-project/data/esp32_data.txt"
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('',4210))

while True:
    data, addr = sock.recvfrom(1024)
    sig = str(data)
    t = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(DATA, 'a') as f:
        f.write(t + ' ' + sig + '\n')
    print(t + ' ' + sig)
