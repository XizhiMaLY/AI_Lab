import argparse
import os
import logging

class Config:
    def __init__(self):
        ...
        self.logger = self._load_logger_n_parse_args()
        
        self.logger.info('\n <<< Config Content <<<\n'+self._read_module_content())
        
        # titanic part
        self.train_data_path = "/mnt/disk1/AI_Lab/to3_titanic/train_titnic.csv"
        self.test_data_path = "/mnt/disk1/AI_Lab/to3_titanic/test_titnic.csv"

        self.prev_val_loss = float('inf')
        self.tolerance = 10

 
    def _read_module_content(self):
        try:
            module_path = __import__(__name__).__file__
            with open(module_path, 'r', encoding='utf-8') as file:
                module_content = file.read()
            return module_content
        except Exception as e:
            print(f"Error occurred while reading module '{module_name}': {e}")
            return None


    def _load_logger_n_parse_args(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        
        parser = argparse.ArgumentParser(description='Your program description')
        parser.add_argument('-r', '--resume', default=None, type=str,
                            help='path to latest checkpoint (default: None)')
        parser.add_argument('-d', '--device', default=None, type=str,
                            help='indices of GPUs to enable (default: all)')
        parser.add_argument('-l', '--log_path', default=None, type=str,
                            help='Just log path')
        parser.add_argument('-lr', '--learning_rate', default=None, type=float,
                            help='')
        parser.add_argument('-e', '--num_epochs', default=None, type=int,
                            help='')
        parser.add_argument('-b', '--batch_size', default=None, type=int,
                            help='')
        args = parser.parse_args()
        
        if args.log_path:
            self.log_path = args.log_path
        else:
            print("Log path is not specified. Please provide a log path using the '-l' or '--log_path' argument.")
            raise NotImplementedError
        
        handler = logging.FileHandler(self.log_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # logger.info('resume_checkpoint: ' + str(args.resume))
        # logger.info('device_indices: '+ str(args.device))
        # logger.info('log_path: ' + str(args.log_path))
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))
            logger.info(f'{arg}: {getattr(args, arg)}')
        return logger
        