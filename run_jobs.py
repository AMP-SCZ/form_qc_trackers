import subprocess
import time

class RunJobs():
    def __init__(self, login, nodes, remote_script):
        self.remote_script = self.remote_script 
        self.login = login
        self.nodes = nodes 
    
    def is_node_available(self, node):
        """
        Checks if the given node is reachable via SSH.
        Assumes `self.login` contains the login string for SSH (e.g., user@hostname).
        """
        try:
            subprocess.check_output(
                ["ssh", self.login, "echo ok"],
                timeout=5
            )
            return True
        except:
            return False
    
    def run_job(self, node):
        """
        Starts the job on the specified node using SSH.
        Returns a subprocess.Popen object so the job can be monitored.
        """
        print(f"Starting job on {node}")
        return subprocess.Popen(
            ["ssh", self.login, self.remote_script]
        )
    
    def main(self):
        """
        Main function that tries each node in the list:
        """
        for node in self.nodes:
            if self.is_node_available(node):  # Check if node is online
                process = self.run_job(node)   # Start the job on that node
                while True:
                    time.sleep(30)  # Wait 30 seconds between checks
                    if process.poll() is not None:  # If job finished
                        print(f"Job on {node} finished.")
                        return
                    if not self.is_node_available(node):  # If node went offline
                        print(f"{node} is unreachable. Restarting job on next node.")
                        process.terminate()  # Stop the current job
                        break  # Move to the next node
        else:
            print("No nodes available to run the job.")  # All nodes failed
