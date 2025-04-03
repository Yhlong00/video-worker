import runpod
import subprocess

def handler(event):
    print("Worker Start")
    input_data = event['input']

    prompt = input_data.get('prompt')
    seconds = input_data.get('seconds', 0)

    print(f"Received prompt: {prompt}")

    command = [
        "python", "generate.py",
        "--task", "i2v-14B",
        "--size", "832*480",
        "--ckpt_dir", "./Wan2.1-I2V-14B-480P",
        "--image", "examples/i2v_input.JPG",
        "--prompt", prompt
    ]

    try:
        process = subprocess.Popen(
            command,
            cwd="/runpod-volume/Wan2.1/",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        full_output = []
        for line in process.stdout:
            print(line.strip())  # Print each line as it comes
            full_output.append(line.strip())

        process.wait()
        if process.returncode != 0:
            return {
                "status": "error",
                "error": "Process exited with error",
                "output": "\n".join(full_output)
            }

        return {
            "status": "success",
            "output": "\n".join(full_output)
        }

    except Exception as e:
        print("Subprocess exception:", str(e))
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})