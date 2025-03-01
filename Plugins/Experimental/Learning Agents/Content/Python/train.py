# -*- coding: utf-8 -*-
'''
Copyright Epic Games, Inc. All Rights Reserved.
'''

import json
import os
import socket
import sys
import traceback
from collections import OrderedDict
from importlib import import_module


def print(*args, **kwargs):
    with open("/Users/a2mogus/iigra/log.txt", 'a') as fd:
        fd.write(' '.join(map(str, args)) + '\n')
        if kwargs:
            fd.write('kwargs ' + str(kwargs) + '\n')
        

if __name__ == '__main__':
    print('Zzzz')
    print("MOP!")


    # TODO: Work out how to make this a little more robust
    sys.path.append(os.path.dirname(__file__) + '/../../../NNERuntimeBasicCpu/Content/Python/')
    
    # Add path where custom trainer files can be found
    if len(sys.argv) <= 1:
        raise Exception('Trainer path not provided')
    else:
        sys.path.append(sys.argv[1])
    
    # Get the name of the trainer to be used
    if len(sys.argv) <= 2:
        raise Exception('Trainer type not provided')
    else:
        trainer_module_name = sys.argv[2]
    
    # Import the module and find the train function
    module = import_module(trainer_module_name)
    train = getattr(module, 'train')

    # Get the communication type
    if len(sys.argv) <= 3:
        raise Exception('Communication type not provided')
    else:
        communication_type = sys.argv[3]
    
    print("Comms!")
    if communication_type == 'SharedMemory':
        
        if len(sys.argv) != 7:
            raise Exception('Wrong number of arguments to Shared Memory Communicator')

        controls_guid = sys.argv[4]
        process_num = int(sys.argv[5])
        config_file = sys.argv[6]

        from train_common import SharedMemoryCommunicator
        communicator = SharedMemoryCommunicator(controls_guid, process_num, config_file)
        
        train(communicator.config, communicator)

        if communicator.config['LoggingEnabled']: print('Exiting...')
        
    elif communication_type == 'Socket':
        print("Socket!")
        
        if len(sys.argv) != 7:
            raise Exception('Wrong number of arguments to Socket Communicator')

        address = sys.argv[4]
        temp_directory = sys.argv[5]
        log_enabled = bool(sys.argv[6])

        print("Argv read!")
        
        from train_common import (
            SocketCommunicator,
            socket_receive_config,
            UE_RESPONSE_SUCCESS)
        
        if log_enabled: print('Starting Socket Communicator...')

        host, port = address.split(':')
        port = int(port)

        log_enabled = True
        
        if log_enabled:
            print('Creating Socket Trainer Server (%s:%i)...' % (host, port))
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        
            if log_enabled: print('Listening...')
            try:
                s.bind((host, port))
            except Exception as e:
                print('No listening!')
                print(e)
                print(e.__repr__())

            print("Bind!")
            s.listen()
            print("Listen!")
            
            while True:
                
                print("Wait for conn!")
                try:
                    conn, addr = s.accept()
                except Exception as e:
                    print(str(e))
                    print(e.__repr__())
                
                print("Get conn!")
                try:
                    
                    if log_enabled: print('Accepted...')

                    with conn:
                        
                        if log_enabled: print('Receiving Config...')
                        
                        response, config = socket_receive_config(conn)
                        
                        if response != UE_RESPONSE_SUCCESS:
                            print("Bad!")
                            print(str(response))
                            print(response.__repr__())
                            raise Exception('Failed to get config...')
                        
                        config = json.loads(config, object_pairs_hook=OrderedDict)
                        config['IntermediatePath'] = temp_directory
                        config['LoggingEnabled'] = log_enabled
                        
                        # Save Config
                        if not os.path.exists(temp_directory+'/Configs/'):
                            os.makedirs(temp_directory+'/Configs/')

                        with open(temp_directory+'/Configs/%s_%s_%s_%s.json' % (
                            config['TaskName'], 
                            config['TrainerMethod'], 
                            communication_type, 
                            config['TimeStamp']), 'w') as f:
                             
                            json.dump(config, f, indent=4)
                        
                        if log_enabled: print('Creating Communicator...')
                        
                        communicator = SocketCommunicator(conn, config)
                        
                        if log_enabled: print('Training...')
                        
                        train(config, communicator)
                        
                    if log_enabled: print('Exiting...')
                    
                except Exception as e:
                    print('Exception During Communication:' + str(e))
                    traceback.print_exc()
    else:
        raise Exception('Unknown Communicator Type %s' % communication_type)
