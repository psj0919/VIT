from model.VIT import ViT
from Config.Config_vit import get_config_dict
from Core.engine_vit import Trainer





if __name__=='__main__':
    cfg = get_config_dict()
    trainer = Trainer(cfg = cfg)
    trainer.training()