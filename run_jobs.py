import subprocess
import time

class RunJobs():
  def __init__(self, login, nodes):
    self.remote_script = ""
    self.login = login
    self.nodes = nodes 
    
  def is_node_available(self,node):
      try:
          subprocess.check_output(
              ["ssh",self.login, "echo ok"],
              timeout=5
          )
          return True
      except:
          return False
  
  def run_job(node):
      print(f"Starting job on {node}")
      return subprocess.Popen(
          ["ssh",self.login, self.remote_Script]
      )
  
  def main(self):
      for node in self.nodes:
          if is_node_available(node):
              process = run_job(node)
              while True:
                  time.sleep(30)
                  if process.poll() is not None:
                      print(f"Job on {node} finished.")
                      return
                  if not is_node_available(node):
                      print(f"{node} is unreachable. Restarting job on next node.")
                      process.terminate()
                      break
      else:
          print("No nodes available to run the job.")
