import random
import socket
import time

# MLLP Framing Characters
SB = b"\x0b"  # <VT> Start Block
EB = b"\x1c"  # <FS> End Block
CR = b"\x0d"  # <CR> Carriage Return


def wrap_mllp(hl7_str: str) -> bytes:
    """Encapsulates the raw HL7 string inside standard MLLP framing."""
    return SB + hl7_str.encode("utf-8") + EB + CR


def generate_vitals_message(msg_id: int) -> str:
    """Generates an ORU^R01 message containing only the requested 6 parameters."""
    ts = time.strftime("%Y%m%d%H%M%S")

    # Dynamic vital signs simulation
    hr = random.randint(65, 100)          # HR: Heart Rate (bpm)
    o2sat = random.randint(95, 100)       # O2Sat: Oxygen Saturation (%)
    sbp = random.randint(110, 135)        # SBP: Systolic Blood Pressure (mmHg)
    dbp = random.randint(70, 85)          # DBP: Diastolic Blood Pressure (mmHg)
    map_val = int((2 * dbp + sbp) / 3)   # MAP: Mean Arterial Pressure (calculated)
    resp = random.randint(12, 20)         # Resp: Respiration Rate (breaths/min)

    # Standard LOINC-coded OBX segments
    segments = [
        f"MSH|^~\\&|SIMULATOR|HOSPITAL|RECEIVER|CLINIC|{ts}||ORU^R01|{msg_id}|P|2.5",
        f"PID|1||PAT0001^^^MRN||Doe^John||19800101|M",
        f"PV1|1|I|ICU^BED1",
        f"OBX|1|NM|8867-4^HR^LN||{hr}|bpm|60-100|N|||F",
        f"OBX|2|NM|2708-6^O2Sat^LN||{o2sat}|%|95-100|N|||F",
        f"OBX|3|NM|8480-6^SBP^LN||{sbp}|mmHg|90-120|N|||F",
        f"OBX|4|NM|8478-0^MAP^LN||{map_val}|mmHg|70-105|N|||F",
        f"OBX|5|NM|8462-4^DBP^LN||{dbp}|mmHg|60-80|N|||F",
        f"OBX|6|NM|9279-1^Resp^LN||{resp}|/min|12-20|N|||F",
    ]
    #hl7 v2

    return "\r".join(segments) + "\r"


def start_sender(host="192.168.1.20", port=2575, interval=1.0):
    print(f"[*] Connecting to HL7 receiver at {host}:{port}...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
        print(f"[+] Connected! Streaming vital parameters. Press Ctrl+C to stop.\n")

        msg_count = 0
        while True:
            msg_count += 1
            hl7_message = generate_vitals_message(msg_count)

            # Send MLLP-wrapped payload
            sock.sendall(wrap_mllp(hl7_message))
            print(f"[{time.strftime('%X')}] Sent packet #{msg_count} (HR, O2Sat, SBP, MAP, DBP, Resp)")

            # Listen for HL7 ACK response
            sock.settimeout(2.0)
            try:
                ack = sock.recv(4096)
                if not ack:
                    print("[-] Receiver closed the connection.")
                    break
            except socket.timeout:
                print(f"[!] Warning: No ACK received for packet #{msg_count}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n[*] Stopped by user. Closing connection...")
    except ConnectionResetError:
        print("\n[-] Receive"
              "r disconnected abruptly.")
    except Exception as e:
        print(f"\n[!] Error: {e}")
    finally:
        sock.close()
        print("[*] Socket closed cleanly.")


if __name__ == "__main__":
    start_sender(host="192.168.1.10", port=2575, interval=1.0)

    # door