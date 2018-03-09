def apply_mode(module, mode):
    if mode == 'initialize' and 'reset_parameters' in dir(module):
        module.reset_parameters()

    for param in module.parameters():
        if mode == 'freeze':
            param.requires_grad = False
        elif mode in ['fine-tune', 'initialize']:
            param.requires_grad = True
