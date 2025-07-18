import os
import subprocess
import time
import json
from pathlib import Path
import shutil

DOCKER_IMAGE = "adobe1a"
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
SAMPLE_PDF = "test_samples/sample_50pages.pdf"  



def run_docker_test():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(SAMPLE_PDF, INPUT_DIR / "test.pdf")

    print("🚀 Running Docker Container...")
    start = time.time()
    subprocess.run([
        "docker", "run", "--rm",
        "-v", f"{os.getcwd()}/input:/app/input",
        "-v", f"{os.getcwd()}/output:/app/output",
        "--network", "none",
        DOCKER_IMAGE
    ], check=True)
    end = time.time()

    runtime = end - start
    print(f"✅ Docker Run Time: {runtime:.2f} seconds")
    return runtime

def check_json_output():
    output_file = OUTPUT_DIR / "test.json"
    if not output_file.exists():
        return False
    with open(output_file) as f:
        try:
            data = json.load(f)
            return "title" in data and "outline" in data
        except json.JSONDecodeError:
            return False

def check_cpu_platform():
    result = subprocess.check_output(["docker", "image", "inspect", DOCKER_IMAGE])
    return b"amd64" in result or b"linux/amd64" in result

def main():
    print("🧪 Running Constraint Tests...\n")

    if not SAMPLE_PDF or not Path(SAMPLE_PDF).exists():
        print("❌ Sample PDF with 50 pages not found at:", SAMPLE_PDF)
        return


    print("1. 🧠 Running & Timing Docker container...")
    runtime = run_docker_test()
    assert runtime <= 10, f"❌ Runtime too long: {runtime:.2f}s (should be ≤ 10s)"
    print("✅ Execution time constraint passed.")

    print("2. 📄 Output JSON Check...")
    assert check_json_output(), "❌ Output JSON not found or invalid."
    print("✅ Output JSON generated correctly.")

    print("3. ⚙️ Platform Check...")
    assert check_cpu_platform(), "❌ Docker image not built for amd64!"
    print("✅ Docker image architecture is amd64.")

    print("\n🎉 All constraints verified successfully!")

if __name__ == "__main__":
    main()
