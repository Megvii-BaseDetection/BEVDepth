from perceptron.engine.cli import BaseCli

from exps.bev_depth_lss_r50_256x704_128x128_24e import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self, **kwargs):
        super(Exp, self).__init__(**kwargs)
        self.key_idxes = [-1]
        self.head_conf['bev_backbone_conf']['in_channels'] = 80 * (
            len(self.key_idxes) + 1)
        self.head_conf['bev_neck_conf']['in_channels'] = [
            80 * (len(self.key_idxes) + 1), 160, 320, 640
        ]
        self.head_conf['train_cfg']['code_weight'] = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]


if __name__ == '__main__':
    BaseCli(Exp).run()
