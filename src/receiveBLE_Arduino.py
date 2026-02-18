import socket
import time

ARDUINO_IP = "192.168.4.1"
PORT = 5000

KICK = b"START\n"   # must send at least 1 byte so Arduino "accepts" us

def receive_forever():
    while True:
        try:
            print(f"Connecting to {ARDUINO_IP}:{PORT} ...")
            s = socket.create_connection((ARDUINO_IP, PORT), timeout=10.0)  # connect timeout only
            s.settimeout(None)  # blocking reads (no read timeout)
            s.sendall(KICK)

            print("Connected. Receiving values...")

            buf = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    raise ConnectionError("socket closed")
                buf += chunk

                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = int(line.decode("ascii", errors="ignore"))
                        print(raw)
                    except ValueError:
                        # e.g. HELLO
                        pass

        except (OSError, ConnectionError) as e:
            print(f"Disconnected ({e}). Reconnecting in 1s...")
            time.sleep(1)

if __name__ == "__main__":
    receive_forever()
