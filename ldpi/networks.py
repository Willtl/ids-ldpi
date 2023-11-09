import functools

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, num_features, rep_dim, bias=True):
        super().__init__()
        self.input_size = input_size
        self.num_features = num_features
        self.rep_dim = rep_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_size, num_features, bias=bias),
            nn.ReLU(),
            nn.Linear(num_features, num_features // 2, bias=bias),
            nn.ReLU(),
            nn.Linear(num_features // 2, rep_dim, bias=bias),
        )

        self.decoder = nn.Sequential(
            nn.Linear(rep_dim, num_features // 2, bias=bias),
            nn.ReLU(),
            nn.Linear(num_features // 2, num_features, bias=bias),
            nn.ReLU(),
            nn.Linear(num_features, input_size, bias=bias)
        )

        # init_weights(self.encoder, init_type='normal')
        # init_weights(self.decoder, init_type='normal')

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))


class Encoder(nn.Module):
    def __init__(self, isize, nz, nc, ndf, n_extra_layers=0, add_final_conv=True):
        super(Encoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf), nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf), nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf), nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf), nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf), nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat), nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat), nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cndf, 1), nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))

        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output


class Decoder(nn.Module):
    def __init__(self, isize, nz, nc, ngf, n_extra_layers=0):
        super(Decoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial-{0}-{1}-convt'.format(nz, cngf), nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf), nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf), nn.ReLU(True))

        csize, _ = 4, cngf
        while csize < isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2), nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2), nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cngf), nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cngf), nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cngf), nn.ReLU(True))

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc), nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        # main.add_module('final-{0}-tanh'.format(nc), nn.Tanh())
        main.add_module('final-{0}-sigmoid'.format(nc), nn.Sigmoid())
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output


class NetG(nn.Module):
    def __init__(self, args):
        super(NetG, self).__init__()
        self.encoder1 = Encoder(args.img_size, args.nz, args.nc, args.ngf, args.extralayers)
        self.decoder = Decoder(args.img_size, args.nz, args.nc, args.ngf, args.extralayers)
        self.encoder2 = Encoder(args.img_size, args.nz, args.nc, args.ngf, args.extralayers)

    def forward(self, x):
        latent_i = self.encoder1(x)
        rec_img = self.decoder(latent_i)
        latent_o = self.encoder2(rec_img)
        return rec_img, latent_i, latent_o


class NetD(nn.Module):
    def __init__(self, args):
        super(NetD, self).__init__()
        model = Encoder(args.img_size, 1, args.nc, args.ndf, args.extralayers)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        # self.features.add_module('ConvFeauture', )
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features


class BasicDiscriminator(nn.Module):
    def __init__(self, args):
        super(BasicDiscriminator, self).__init__()
        img_size = args.img_size
        # nz = args.nz * 10
        nz = 100
        nc = args.nc
        ndf = args.ndf
        n_extra_layers = 0

        assert img_size % 16 == 0, "img_size has to be a multiple of 16"

        feat = nn.Sequential()
        clas = nn.Sequential()
        # input is nc x img_size x img_size
        feat.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        feat.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = img_size / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            feat.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            feat.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            feat.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            feat.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            feat.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            feat.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        # main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
        #                     nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))
        feat.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                        nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))
        clas.add_module('classifier', nn.Conv2d(nz, 1, 3, 1, 1, bias=False))
        clas.add_module('Sigmoid', nn.Sigmoid())

        self.feat = feat
        self.clas = clas

    def forward(self, x):
        feat = self.feat(x)
        clas = self.clas(feat)
        clas = clas.view(-1, 1).squeeze(1)
        return clas, feat


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # Construct U-NET structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, x):
        return self.model(x)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class Autoencoder(nn.Module):
    def __init__(self, args):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(args.img_size, args.nz, args.nc, args.ngf, args.extralayers)
        self.decoder = Decoder(args.img_size, args.nz, args.nc, args.ngf, args.extralayers)

    def forward(self, x):
        latent = self.encoder(x)
        rec_img = self.decoder(latent)
        return latent, rec_img


class AAEDiscriminator(nn.Module):
    def __init__(self, args):
        super(AAEDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(args.nz, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(64, 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.model(z)


def create_generator(args):
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    num_layer = int(np.log2(args.img_size))
    net_g = UnetGenerator(args.nc, args.nc, num_layer, args.ngf, norm_layer=norm_layer, use_dropout=False)
    init_weights(net_g, init_type='normal')
    return net_g


def create_discriminator(args):
    net_d = BasicDiscriminator(args)
    init_weights(net_d, init_type='normal')
    return net_d


def print_net_summary(net_name, net, input_shape):
    print(f'\n\n{net_name}')
    summary(net, input_shape)
    print('\n\n')


def initialize_weights(model):
    print('initialize weights')
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.zeros_(m.bias.data)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        with torch.no_grad():
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
                torch.nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
