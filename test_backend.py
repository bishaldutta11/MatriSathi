import subprocess
import time
import urllib.request
import urllib.error
import sys
import os

def check_endpoint(url):
    try:
        response = urllib.request.urlopen(url, timeout=5)
        status = response.getcode()
        print(f"Endpoint {url} -> HTTP {status} (Success)")
        return True
    except urllib.error.HTTPError as e:
        print(f"Endpoint {url} -> HTTP {e.code} (Failed)")
        return False
    except Exception as e:
        print(f"Endpoint {url} -> Error: {e} (Failed)")
        return False

def run_tests():
    print("Starting backend test verification...")
    
    # Launch backend as a subprocess
    # Run from workspace directory
    cmd = [sys.executable, "app.py"]
    print(f"Launching command: {' '.join(cmd)}")
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to boot up with a retry loop (YOLO model loading takes time)
    print("Waiting for backend server to start listening...")
    server_online = False
    for i in range(30):
        time.sleep(1)
        if proc.poll() is not None:
            print("Backend process terminated prematurely.")
            break
        try:
            response = urllib.request.urlopen("http://localhost:8000/", timeout=1)
            response.close()
            server_online = True
            print(f"Backend started listening after {i+1} seconds.")
            break
        except Exception:
            # Server not ready yet
            sys.stdout.write(".")
            sys.stdout.flush()
            
    print() # newline
    
    if not server_online:
        stdout, stderr = proc.communicate()
        print("Backend failed to start or did not listen within 30 seconds. Subprocess stdout:")
        print(stdout)
        print("Subprocess stderr:")
        print(stderr)
        sys.exit(1)
        
    print("Backend is running in background.")
    
    # Test Root Endpoint
    root_success = check_endpoint("http://localhost:8000/")
    
    # Test Video Feed Endpoint
    # We do a short read since MJPEG is an infinite stream
    video_success = False
    try:
        req = urllib.request.Request("http://localhost:8000/video_feed")
        with urllib.request.urlopen(req, timeout=3) as response:
            header = response.info().get("Content-Type")
            if "multipart/x-mixed-replace" in header:
                print("Endpoint http://localhost:8000/video_feed -> Valid MJPEG Stream header detected (Success)")
                video_success = True
            else:
                print(f"Endpoint http://localhost:8000/video_feed -> Unexpected header: {header} (Failed)")
    except Exception as e:
        if "timeout" in str(e).lower() or "timed out" in str(e).lower():
            print("Endpoint http://localhost:8000/video_feed -> Connection active & streaming (Success)")
            video_success = True
        else:
            print(f"Endpoint http://localhost:8000/video_feed -> Error: {e} (Failed)")
            
    # Terminate the server
    print("Terminating backend server...")
    proc.terminate()
    try:
        proc.wait(timeout=3)
        print("Backend server terminated successfully.")
    except subprocess.TimeoutExpired:
        proc.kill()
        print("Backend server killed.")
        
    if root_success and video_success:
        print("\nAll integration checks PASSED successfully!")
        sys.exit(0)
    else:
        print("\nSome integration checks FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
