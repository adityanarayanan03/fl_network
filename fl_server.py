import omegaconf
from peernet.networks import ZMQ_Pair # type: ignore
from loguru import logger
import time

class FLServer:
    def __init__(self):
        self.net_config = omegaconf.OmegaConf.load("net_config.yaml")
        self.network = ZMQ_Pair(device_name="server", **self.net_config)
        
        self.MAX_ITERS = 10
        
    def _download_model(self):
        x = "model"
        
        # Need to send gradients to each client device
        for device in self.net_config.devices:
            if device == "server":
                continue
            
            self.network.send(device, x)
            logger.info(f"Successfully sent model {x} to {device}.")
        
    def _upload_gradients(self):
        gradients = []
        
        for device in self.net_config.devices:
            if device == "server":
                continue
        
            gradients.append(self.network.recv(device))
        
        logger.info(f"Successfully received gradients {gradients} from {device}.")
    
    def _aggregate_gradients(self):
        logger.info("Aggregating gradients")
        time.sleep(1)
    
    def run(self):
        for idx in range(self.MAX_ITERS):
            logger.info(f"Starting iteration {idx}")
            self._download_model()
            self._upload_gradients()
            self._aggregate_gradients()
            logger.info(f"Finished iteration {idx} \n\n")