import subprocess


def run_dvc_command(command):
    """Run a DVC command and return the output."""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with error: {e.stderr}")
        return None


def save_dvc_dag_as_png(output_path="dag_visualization.png"):
    """Generate the DVC DAG and save it as a PNG."""
    try:
        # Generate the DAG in DOT format
        dot_output = run_dvc_command("dvc dag --dot")
        if not dot_output:
            print("No DAG data available. Please ensure DVC is installed and initialized.")
            return

        # Save the DOT output to a temporary file
        with open("dag.dot", "w") as dot_file:
            dot_file.write(dot_output)

        # Use Graphviz to convert the DOT file to a PNG
        subprocess.run(f"dot -Tpng dag.dot -o {output_path}", shell=True, check=True)
        print(f"DAG visualization saved as {output_path}")

    except Exception as e:
        print(f"Error: Failed to save DAG visualization. {str(e)}")


def main():
    print("Generating and saving DVC DAG visualization...")
    save_dvc_dag_as_png()


if __name__ == "__main__":
    main()