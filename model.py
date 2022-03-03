from transformers import AutoModel, AutoConfig
import torch.nn as nn
from torch.cuda.amp import autocast

class Encoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.config= AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config= self.config)
        self.backbone.init_weights()

    @autocast()
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooler_output = outputs['pooler_output']  
        

        return pooler_output

class DPR(nn.Module):
    def __init__(self, q_model_name, p_model_name):
        super().__init__()
        
        self.q_model= Encoder(q_model_name)
        self.p_model= Encoder(p_model_name)
    
    def _load_model(self):
        return self.q_model, self.p_model