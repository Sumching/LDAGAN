from network.sngan_net import ResNetGenerator, ResNetDiscriminator
from network.vgan_128_net import VGAN128Generator, VGAN128Discriminator
from network.vgan_1024_net import VGAN1024Generator, VGAN1024Discriminator

__all__ = ['defineNet']

def defineNet(image_size=32):
    """Choose different models according to different sizes"""
    if image_size == 32:
        return ResNetGenerator, ResNetDiscriminator
    elif image_size == 128:
        return VGAN128Generator, VGAN128Discriminator
    elif image_size == 1024:
        return VGAN1024Generator, VGAN1024Discriminator
    else:
        raise NotImplementedError