import paramiko

# SSH connection details
host = '192.168.126.131'
port = 22  # Default SSH port
username = 'rahul'
password = 'rahul'

# Create SSH client
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Connect to the remote VM
ssh.connect(host, port, username, password)

# Execute a Python script on the remote machine

remote_script_path = '/home/rahul/Downloads/ms-identity-python-webapp-master'
command = f'cd /home/rahul/Downloads/ms-identity-python-webapp-master;python3 execute.py'
stdin, stdout, stderr = ssh.exec_command(command)

# Retrieve and print the output
output = stdout.read().decode('utf-8')
print(output)
print(stderr)

# Close the SSH connection
ssh.close()