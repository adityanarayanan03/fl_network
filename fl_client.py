import omegaconf
from peernet.networks import ZMQ_Pair # type: ignore
import time
from loguru import logger

class FLClient:
    def __init__(self, device_name:str):
        self.net_config = omegaconf.OmegaConf.load("net_config.yaml")
        self.network = ZMQ_Pair(device_name=device_name, **self.net_config)
        
        self.MAX_ITERS = 10
        
    def _download_model(self):
        x = self.network.recv("server")
        logger.info(f"Successfully received model {x} from server.")
        return x
    
    def _train(self):
        logger.info("Training model")
        time.sleep(1)
    
    def _upload_gradients(self):
        gradients = "gradients"
        self.network.send("server", gradients)
        logger.info(f"Successfully sent gradients {gradients} to server.")
        
    def run(self):
        for idx in range(self.MAX_ITERS):
            logger.info(f"Starting iteration {idx}")
            self._download_model()
            self._train()
            self._upload_gradients()
            logger.info(f"Finished iteration {idx} \n\n")