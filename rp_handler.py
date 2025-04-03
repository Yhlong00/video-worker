import runpod
import subprocess

def handler(event):
    print(f"Worker Start")
    input_data = event['input']

    prompt = input_data.get('prompt')
    seconds = input_data.get('seconds', 0)

    print(f"Received prompt: {prompt}")

    # Define command
    command = [
        "python", "generate.py",
        "--task", "i2v-14B",
        "--size", "832*480",
        "--ckpt_dir", "./Wan2.1-I2V-14B-480P",
        "--image", "examples/i2v_input.JPG",
        "--prompt", prompt
    ]

    try:
        # Run the command in /runpod-volume/folder/
        result = subprocess.run(
            command,
            cwd="/runpod-volume/Wan2.1/",
            capture_output=True,
            text=True,
            check=True
        )
        print("Command output:", result.stdout)
        return {
            "status": "success",
            "output": result.stdout.strip()
        }
    except subprocess.CalledProcessError as e:
        print("Command failed:", e.stderr)
        return {
            "status": "error",
            "error": e.stderr.strip()
        }

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})