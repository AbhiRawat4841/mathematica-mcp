import subprocess
import json
import sys

def run_test():
    # Command to start the server
    cmd = ["uv", "run", "mathematica-mcp"]
    
    # Start process
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True
    )

    # JSON-RPC Initialize Request
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"}
        }
    }

    print(f"Sending: {json.dumps(init_request)}")
    
    try:
        # Write to stdin
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        
        # Read response
        response = process.stdout.readline()
        print(f"Received: {response}")
        
        if "result" in response and "capabilities" in response:
            print("SUCCESS: Server handshake completed.")
        else:
            print("FAILURE: Invalid response.")
            
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        process.kill()

if __name__ == "__main__":
    run_test()
