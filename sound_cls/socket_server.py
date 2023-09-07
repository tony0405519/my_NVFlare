import socket
import sys
import os

def recv_to_newline(s):
    buf = []
    while True:
        c = s.recv(1)
        if not len(c):
            # socket closed
            return None
        if c == b"\n":
            return b"".join(buf)
        buf.append(c)

SERVER_PORT = 8104
BUFFER_SIZE = 4096
SERVER_HOST = '0.0.0.0'
error = False

s = socket.socket()
s.bind((SERVER_HOST, SERVER_PORT))

while True:
    s.listen(5)
    client, address = s.accept()

    filename = recv_to_newline(client).decode("utf-8")
    print(filename)
    filesize = int(recv_to_newline(client).decode("utf-8"))
    print(filesize)

    with open(filename, "wb") as f:
        while filesize:
            bytes_read = client.recv(min(1024, filesize))
            if not bytes_read:
                error = True
                break
            f.write(bytes_read)
            filesize -= len(bytes_read)
    if error:
        os.remove(filename)
    client.close()
    print(f'save a new model from server !!!')
s.close()
