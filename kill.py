##########
#Author: Yu Chao
#data: 2022/4/15
#Read the PID of the port and close.
#Project: mmwave passive bistatic radar 
import os


def killport(port):
    command = 'sudo lsof -i:' + str(port)
    password = '1112'
    M = os.popen('echo %s | sudo -S %s' % (password,command)).readlines()
    PID = M[1].split(' ')
    for i in PID:
        if i == '' or i == ' ':
            PID.remove(i)
    kill_cmd = 'sudo kill ' + PID[1]
    os.system(kill_cmd)
    print('The port of %s was killed',port)

def main():
    ref_port = 5220
    tar_port = 5222
    killport(ref_port)
    killport(tar_port)


if __name__ == '__main__':
    main()
    
    
