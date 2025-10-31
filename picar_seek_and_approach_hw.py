import os, time, pathlib, threading, math, requests, json
from time import strftime, localtime
from robot_hat.utils import reset_mcu
from picarx import Picarx
from vilib import Vilib
from PIL import Image

# ===================== CONFIG =====================
LAPTOP_IP   = os.getenv("LAPTOP_IP", "172.20.10.13")
API_PORT    = int(os.getenv("API_PORT", "8000"))
API_URL     = f"http://{LAPTOP_IP}:{API_PORT}/predict"

TARGET_LABELS = os.getenv("TARGET_LABELS", "caterpillar")  # e.g., "person,cat"; empty = any
CONF_THRESH   = float(os.getenv("CONF_THRESH", "0.5"))

# Scan behavior
SCAN_SPEED        = int(os.getenv("SCAN_SPEED", "20"))    # drive speed while scanning
SCAN_STEER_DEG    = int(os.getenv("SCAN_STEER_DEG", "-20")) # left circle (negative left)
PAN_MIN, PAN_MAX  = -60, 60
PAN_STEP_DEG      = 20
PAN_DWELL_SEC     = 0.05

# Approach behavior
APPROACH_SPEED    = int(os.getenv("APPROACH_SPEED", "70"))
STEER_KP          = float(os.getenv("STEER_KP", "1"))   # gain mapping bearing -> steering
MAX_STEER_DEG     = 30

# Termination / safety heuristics
AREA_NEAR_THRESH  = float(os.getenv("AREA_NEAR_THRESH", "0.3"))  # relative box area considered "close"
LOST_TIMEOUT_SEC  = float(os.getenv("LOST_TIMEOUT_SEC", "1"))  # time if robot lost target
SNAPSHOT_INTERVAL = float(os.getenv("SNAPSHOT_INTERVAL", "0.2")) # seconds between snapshots

# Where Vilib stores photos
PHOTO_DIR = os.path.join(os.path.expanduser(f'~{os.getlogin()}'), "Pictures", "picar-x")
pathlib.Path(PHOTO_DIR).mkdir(parents=True, exist_ok=True)
# ==================================================

def vilib_take_photo(path_dir=PHOTO_DIR):
    ts = strftime('%Y-%m-%d-%H-%M-%S', localtime(time.time()))
    name = f'photo_{ts}'
    Vilib.take_photo(name, path_dir) # Which function to take phote
    return os.path.join(path_dir, f"{name}.jpg")

def post_image_to_api(jpg_path):
    with open(jpg_path, "rb") as f:
        r = requests.post(API_URL, files={"image": (os.path.basename(jpg_path), f, "image/jpeg")}, timeout=5)
    r.raise_for_status()
    return r.json()

def pick_target(pred_json, prefer_labels):
    """
    pred_json = {"prediction":[{"label":"person","score":0.87,"bbox":[x1,y1,x2,y2]}, ...]}
    Return dict with best target or None.
    """
    dets = pred_json.get("prediction", []) or []
    if not dets:
        return None
    # Filter by labels if provided
    if prefer_labels:
        wanted = [d for d in dets if d.get("label","") in prefer_labels and d.get("score",0) >= CONF_THRESH]
        pool = wanted if wanted else [d for d in dets if d.get("score",0) >= CONF_THRESH]
    else:
        pool = [d for d in dets if d.get("score",0) >= CONF_THRESH]
    if not pool:
        return None
    # Pick largest area (tends to be closest)
    def area(d):
        x1,y1,x2,y2 = d["bbox"]
        return max(0.0,(x2-x1)) * max(0.0,(y2-y1))
    pool.sort(key=area, reverse=True)
    return pool[0]

def bbox_center_and_area(bbox, img_w, img_h):
    x1,y1,x2,y2 = bbox
    cx = (x1+x2)/2.0  
    cy = (y1+y2)/2.0        # How to find center x and ceter y? 
    area = max(0.0,(x2-x1)) * max(0.0,(y2-y1))
    rel_area = area / float(img_w*img_h)
    return cx, cy, rel_area

def clamp(v, lo, hi): return max(lo, min(hi, v))

def main():
    prefer_labels = [s.strip() for s in TARGET_LABELS.split(",") if s.strip()]
    print(f"[INFO] Using API: {API_URL}")
    print(f"[INFO] Target labels: {prefer_labels if prefer_labels else 'ANY'} (conf>={CONF_THRESH})")

    reset_mcu(); time.sleep(0.2)
    px = Picarx()

    # Start Vilib camera UI
    Vilib.camera_start(vflip=False, hflip=False)       # How to start camera?
    Vilib.display(local=True, web=True)
    time.sleep(1.5)

    # State             # SCAN -> APPROACH --> LOST --> SCAN
    mode = "SCAN"    
    last_seen_time = 0
    cur_pan = 0
    pan_dir = +1

    try:
        while True:
            loop_t0 = time.time()

            # 1) Take snapshot via Vilib using previously defined function
            jpg_path = vilib_take_photo(path_dir=PHOTO_DIR)

            # 2) Send to API & parse
            target = None
            try:
                resp = post_image_to_api(jpg_path)        # Post image to API using previously defined function
                target = pick_target(resp, prefer_labels)          # Use previously defined function to find target
            except Exception as e:
                print("[WARN] POST failed:", e)

            # 3) Decide behavior
            if target:
                last_seen_time = time.time()
                # Open image to get size for relative area calc
                try:
                    with Image.open(jpg_path) as im:
                        w, h = im.size
                except:
                    w, h = 640, 480  # fallback
                cx, cy, rel_area = bbox_center_and_area(target["bbox"], w, h)

                # The camera is panned at cur_pan (deg). Use it as bearing proxy.
                # Optionally bias by pixel-offset from center for finer aim:
                offset_px = (cx - (w/2)) / (w/2)  # -1 .. +1
                bearing_deg = cur_pan + offset_px * 20.0  # 20° ~ rough horiz FOV contribution

                steer_cmd = clamp(int(STEER_KP * bearing_deg), -MAX_STEER_DEG, MAX_STEER_DEG) # Calculate the steering command using steering control gain and object bearing angle (K * ang)

                if mode != "APPROACH":
                    print("\n[STATE] Lock → APPROACH")
                    mode = "APPROACH"

                # Steering & motion
                px.set_dir_servo_angle(steer_cmd)       # How to set directional servo (Steering servo) angle?
                if rel_area >= AREA_NEAR_THRESH:
                    # close enough → stop and re-center camera
                    print(f"[APPROACH] CLOSE: area={rel_area:.2f} steer={steer_cmd} label={target.get('label')}")
                    px.stop()
                    # put cam near straight for next iteration
                    cur_pan = 0
                    px.set_cam_pan_angle(cur_pan)
                    time.sleep(0.2)
                else:
                    print(f"[APPROACH] area={rel_area:.3f} steer={steer_cmd} label={target.get('label')} conf={target.get('score'):.2f}")
                    px.forward(APPROACH_SPEED)

            else:
                # No target
                if (time.time() - last_seen_time) > LOST_TIMEOUT_SEC:
                    if mode != "SCAN":
                        print("\n[STATE] Lost target → SCAN")
                        mode = "SCAN"

                if mode == "SCAN":
                    # slow left circle + pan sweep
                    px.set_dir_servo_angle(SCAN_STEER_DEG)
                    px.forward(SCAN_SPEED)

                    # pan sweep
                    px.set_cam_pan_angle(cur_pan)
                    cur_pan += pan_dir * PAN_STEP_DEG
                    if cur_pan >= PAN_MAX:
                        cur_pan = PAN_MAX; pan_dir = -1
                    elif cur_pan <= PAN_MIN:
                        cur_pan = PAN_MIN; pan_dir = +1    # How to switch robot camera pan direction when reach max min angle?

            # pacing
            dt = time.time() - loop_t0
            sleep_left = max(0.0, SNAPSHOT_INTERVAL - dt)
            time.sleep(sleep_left)

    except KeyboardInterrupt:
        print("\n[EXIT] KeyboardInterrupt")
    finally:
        px.stop()
        Vilib.camera_close()
        time.sleep(0.2)

if __name__ == "__main__":
    main()
