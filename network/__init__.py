from network.sngan_net import ResNetGenerator, ResNetDiscriminator

__all__ = ['defineNet']

def defineNet(image_size=32):
    """Choose different models according to different sizes"""
    if image_size == 32:
        return ResNetGenerator, ResNetDiscriminator
    elif image_size == 128:
        pass
    elif image_size == 1024:
        pass
    else:
        raise NotImplementedError
