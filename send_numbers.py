"""
连接 192.168.1.20:55555，向设备发送数字 1-10。
"""

import socket
import struct

HOST = "192.168.1.20"
PORT = 55555


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(5.0)
        sock.connect((HOST, PORT))
        print(f"已连接到 {HOST}:{PORT}")

        for n in range(1, 11):
            data = struct.pack("<I", n)
            sock.sendall(data)
            print(f"已发送: {n}")

    print("发送完成，连接已关闭")


if __name__ == "__main__":
    main()
