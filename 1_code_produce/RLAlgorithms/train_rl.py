
import argparse
import os
import subprocess
import sys


def main():

    parser = argparse.ArgumentParser( #create command line parser
        description="Start one RL training script"
    )

    parser.add_argument(
        "--algo",
        type = str,
        required = True,
        choices = ["ppo", "apexdqn", "a3c"],
        help = "Which RL algorithm do you wannt to train?"
    )

    args = parser.parse_args() #parse arguments 

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    algo_to_file = { #maps command line input to actual file 
        "ppo": "PPO.py",
        "apexdqn": "APEXDQN.py",
        "a3c": "A3C.py",
    }

    script_name = algo_to_file[args.algo]
    script_path = os.path.join(base_dir, "RLAlgorithms", script_name)

    if not os.path.isfile(script_path):
        raise FileNotFoundError(
            f"Could not find training script: {script_path}"
        )

    #simply for debugging 
    print(f"Starting training for algorithm: {args.algo}")
    print(f"Using script: {script_path}")

    # We call the existing algorithm file (for example APEXDQN.py) as a separate Python process.
    # This means each algo remains responsible for its own setup and logging 

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=base_dir, #current working directory 
        check=False, 
    )

    if result.returncode != 0: #after the training, returncode == 0 means everything went well, otherwise there was a problem 
        raise RuntimeError(
            f"Training script {script_name} exited with return code {result.returncode}"
        )

    print(f"Training for {args.algo} finished successfully.")


if __name__ == "__main__":
    main()