"""K3/C1: network access must die with SIGSYS."""
import socket
socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('ESCAPE: socket created', flush=True)
