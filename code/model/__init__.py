def get_model(save_manager):
    from model.model import IKEM as Model
    
    model = Model(save_manager.config)
    return model
