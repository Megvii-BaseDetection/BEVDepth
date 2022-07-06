from perceptron.engine.cli import BaseCli

from exps.bev_depth_lss_r50_256x704_128x128_20e_cbgs_key4_pcfe import \
    Exp as BaseExp


class Exp(BaseExp):
    def __init__(self, **kwargs):
        super(Exp, self).__init__(**kwargs)
        scale = 2
        self.backbone_conf.update({'final_dim': (256 * scale, 704 * scale)})
        self.ida_aug_conf.update({
            'final_dim': (256 * scale, 704 * scale),
            'resize_lim': (0.386 * scale, 0.55 * scale)
        })
        self.backbone_conf.update({'x_bound': [-51.2, 51.2, 0.4]})
        self.backbone_conf.update({'y_bound': [-51.2, 51.2, 0.4]})
        self.head_conf['bbox_coder'].update({'out_size_factor': 2})
        self.head_conf['train_cfg'].update({'out_size_factor': 2})
        self.head_conf['test_cfg'].update({'out_size_factor': 2})


if __name__ == '__main__':
    BaseCli(Exp).run()
