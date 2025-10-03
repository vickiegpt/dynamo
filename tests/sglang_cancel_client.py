# test_abort_minimal.py
import json, requests, threading, time

BASE = "http://127.0.0.1:30001"   # point to the worker (or router that forwards /abort_request)
GEN_URL = f"{BASE}/generate"
ABORT_URL = f"{BASE}/abort_request"

A = {"qid": None, "got_token": False, "response": "", "finish_reason": None}
B = {"qid": None, "got_token": False, "response": "", "finish_reason": None}

def stream(name, state, prompt):
    headers = {"Accept": "text/event-stream"}
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": 20000, "temperature": 0.2},
        "stream": True,
    }
    with requests.post(GEN_URL, json=payload, headers=headers, stream=True, timeout=600) as r:
        r.raise_for_status()
        printed_qid = False
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            try:
                obj = json.loads(line)
            except Exception:
                continue

            meta = obj.get("meta_info") or {}
            qid = meta.get("id")
            if qid and not state["qid"]:
                state["qid"] = qid
            if qid and not printed_qid:
                printed_qid = True
                print(f"{time.strftime('%H:%M:%S')} [{name}] qid={qid}")

            tok = obj.get("text_delta") or obj.get("text")
            if tok:
                state["got_token"] = True
                state["response"] += tok

            fr = meta.get("finish_reason")
            if isinstance(fr, dict):
                state["finish_reason"] = fr
                print(f"{time.strftime('%H:%M:%S')} [{name}] finish_reason={fr}")
                break

def abort_A_when_ready():
    # wait until A has a qid AND has started streaming
    deadline = time.time() + 30
    while (not A["qid"] or not A["got_token"]) and time.time() < deadline:
        time.sleep(0.05)
    if not A["qid"]:
        print(f"{time.strftime('%H:%M:%S')} [ABORT->A] no qid; abort not sent")
        return
    time.sleep(1.0)  # small grace so A is mid-decode
    payload = {"rid": A["qid"]}  # Use 'rid' instead of 'qid' for SGLang
    resp = requests.post(ABORT_URL, json=payload, timeout=30)
    try:
        body = resp.json()
    except Exception:
        body = resp.text
    print(f"\n{time.strftime('%H:%M:%S')} [ABORT->A] {resp.status_code} {payload} resp={repr(body)}")

def main():
    # Simple distinct prompts for easy identification
    prompt_A = "Weather: sunny cloudy rainy windy stormy"
    prompt_B = "Music: jazz rock pop blues classical"

    tA = threading.Thread(target=stream, args=("A", A, prompt_A), daemon=True)
    tB = threading.Thread(target=stream, args=("B", B, prompt_B), daemon=True)
    tA.start(); tB.start()

    threading.Thread(target=abort_A_when_ready, daemon=True).start()

    tA.join(); tB.join()
    
    # Print final responses
    print(f"\n{time.strftime('%H:%M:%S')} [FINAL RESPONSES]")
    print(f"[A] Response: {repr(A['response'])}")
    print(f"[B] Response: {repr(B['response'])}")
    
    # Test validation
    a_finish_type = A.get("finish_reason", {}).get("type") if A.get("finish_reason") else None
    b_finish_type = B.get("finish_reason", {}).get("type") if B.get("finish_reason") else None
    
    print(f"\n{time.strftime('%H:%M:%S')} [TEST VALIDATION]")
    print(f"A finish reason type: {a_finish_type}")
    print(f"B finish reason type: {b_finish_type}")
    
    if a_finish_type == "abort" and b_finish_type == "stop":
        print("✅ TEST PASSED: A was aborted, B completed normally")
        exit(0)
    else:
        print("❌ TEST FAILED: Expected A to be 'abort' and B to be 'stop'")
        exit(1)

if __name__ == "__main__":
    main()
