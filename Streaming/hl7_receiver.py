import socket
import sys
from datetime import datetime
import predictor

# --- MLLP Framing Characters ---
START_BLOCK = b'\x0b'     # ASCII 11 (VT)
END_BLOCK = b'\x1c\x0d'   # ASCII 28 (FS) + ASCII 13 (CR)

# Standard LOINC map for telemetry vitals
LOINC_MAP = {
    "8867-4": "HR",
    "2708-6": "O2Sat",
    "8480-6": "SBP",
    "8478-0": "MAP",
    "8462-4": "DBP",
    "9279-1": "Resp"
}


def create_hl7_ack(msh_segment: str) -> str:
    """
    Generates a standard HL7 v2.3 ACK response matching sender MSH details.
    """
    fields = msh_segment.split("|")
    sending_app = fields[2] if len(fields) > 2 and fields[2] else "SENDER"
    sending_facility = fields[3] if len(fields) > 3 and fields[3] else "FACILITY"
    msg_control_id = fields[9] if len(fields) > 9 and fields[9] else "MSG00001"

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    ack_lines = [
        f"MSH|^~\\&|HL7_RECEIVER|ICU_SERVER|{sending_app}|{sending_facility}|{timestamp}||ACK^R01|ACK{timestamp}|P|2.3",
        f"MSA|AA|{msg_control_id}"  # AA = Application Accept
    ]
    return "\r".join(ack_lines) + "\r"


def parse_hl7_message(raw_msg: str):
    """
    Extracts PID-3 and parses all OBX numerical segments.
    """
    segments = [s.strip() for s in raw_msg.split("\r") if s.strip()]

    patient_id = "UNKNOWN"
    vitals = {}
    msh_segment = ""

    for segment in segments:
        fields = segment.split("|")
        seg_type = fields[0]

        if seg_type == "MSH":
            msh_segment = segment

        elif seg_type == "PID":
            if len(fields) > 3:
                # Handle subcomponents if present (e.g. ID^^^Hospital)
                patient_id = fields[3].split("^")[0]

        elif seg_type == "OBX":
            if len(fields) > 5:
                loinc_info = fields[3].split("^")
                loinc_code = loinc_info[0]
                raw_value = fields[5]
                units = fields[6] if len(fields) > 6 else ""

                try:
                    num_val = float(raw_value) if "." in raw_value else int(raw_value)
                except ValueError:
                    num_val = raw_value

                metric_name = LOINC_MAP.get(
                    loinc_code,
                    loinc_info[1] if len(loinc_info) > 1 and loinc_info[1] else loinc_code
                )
                vitals[metric_name] = {
                    "value": num_val,
                    "units": units,
                    "loinc": loinc_code
                }

    return msh_segment, patient_id, vitals


def start_receiver(host: str = "0.0.0.0", port: int = 2575):
    """
    Listens on TCP, handles multi-message MLLP streams, and returns ACKs.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((host, port))
        server_sock.listen(5)
        print(f"[*] HL7 Receiver running on {host}:{port} (listening for MLLP)...")

        try:
            while True:
                conn, addr = server_sock.accept()
                with conn:
                    conn.settimeout(15.0)
                    buffer = bytearray()

                    while True:
                        try:
                            chunk = conn.recv(4096)
                            if not chunk:
                                break
                            buffer.extend(chunk)

                            # Process all complete MLLP frames currently in buffer
                            while True:
                                start_idx = buffer.find(START_BLOCK)
                                end_idx = buffer.find(END_BLOCK)

                                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                                    raw_hl7_bytes = buffer[start_idx + 1:end_idx]
                                    del buffer[:end_idx + len(END_BLOCK)]

                                    raw_hl7_text = raw_hl7_bytes.decode("utf-8", errors="replace")
                                    msh_segment, patient_id, vitals = parse_hl7_message(raw_hl7_text)

                                    # Run low-latency XGBoost inference & Native TreeSHAP explainability
                                    predictor.predict_and_explain(patient_id, vitals)

                                    # Send HL7 ACK back to monitor
                                    ack_hl7 = create_hl7_ack(msh_segment)
                                    conn.sendall(START_BLOCK + ack_hl7.encode("utf-8") + END_BLOCK)
                                else:
                                    break

                        except socket.timeout:
                            break

        except KeyboardInterrupt:
            print("\n[*] Receiver stopped by user.")
            sys.exit(0)


if __name__ == "__main__":
    start_receiver(host="0.0.0.0", port=2575)