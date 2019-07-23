import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import itertools

class GestureGANOneCycleModel(BaseModel):
    def name(self):
        return 'GestureGANOneCycleModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='instance')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(which_model_netG='resnet_9blocks')
        parser.add_argument('--REGULARIZATION', type=float, default=1e-6)
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--cyc_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
            parser.add_argument('--lambda_identity', type=float, default=5.0, help='weight for identity loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN_D1',  'G_GAN_D2', 'G_L1', 'G_VGG', 'reg', 'G' , 'D1_real', 'D1_fake','D1', 'D2_real', 'D2_fake','D2']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.opt.saveDisk:
            self.visual_names = ['real_A', 'fake_B', 'real_B']
        else:
            self.visual_names = ['real_A', 'real_D', 'fake_B', 'real_B', 'real_C', 'recovery_A', 'identity_A']
        
        if self.isTrain:
            self.model_names = ['G','D1','D2']
        else:  # during test time, only load G
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(6, 3, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD1 = networks.define_D(6, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD2 = networks.define_D(9, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD1.parameters(),self.netD2.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_C = input['C'].to(self.device)
        self.real_D = input['D'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        combine_AD=torch.cat((self.real_A, self.real_D), 1)
        self.fake_B = self.netG(combine_AD)
        combine_BC=torch.cat((self.fake_B, self.real_C), 1)
        self.recovery_A = self.netG(combine_BC)

        combine_AC=torch.cat((self.real_A, self.real_C), 1)
        self.identity_A = self.netG(combine_AC)


    def backward_D1(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_D1_fake = self.netD1(fake_AB.detach())
        self.loss_D1_fake = self.criterionGAN(pred_D1_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real_D1 = self.netD1(real_AB)
        self.loss_D1_real = self.criterionGAN(pred_real_D1, True)

        # Combined loss
        self.loss_D1 = (self.loss_D1_fake + self.loss_D1_real) * 0.5

        self.loss_D1.backward()

    def backward_D2(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_ADB = self.fake_AB_pool.query(torch.cat((self.real_A, self.real_D, self.fake_B), 1))
        pred_D2_fake = self.netD2(fake_ADB.detach())
        self.loss_D2_fake = self.criterionGAN(pred_D2_fake, False)

        # Real
        real_ADB = torch.cat((self.real_A, self.real_D, self.real_B), 1)
        pred_real_D2 = self.netD2(real_ADB)
        self.loss_D2_real = self.criterionGAN(pred_real_D2, True)

        # Combined loss
        self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) * 0.5

        self.loss_D2.backward()


    def backward_G(self):
        # fake_B should fake the discriminator D1
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_D1_fake = self.netD1(fake_AB)
        self.loss_G_GAN_D1 = self.criterionGAN(pred_D1_fake, True)

        # fake_B should fake the discriminator D2
        fake_ADB = torch.cat((self.real_A, self.real_D, self.fake_B), 1)
        pred_D2_fake = self.netD2(fake_ADB)
        self.loss_G_GAN_D2 = self.criterionGAN(pred_D2_fake, True)

        # color loss
        self.fake_B_red = self.fake_B[:,0:1,:,:]
        self.fake_B_green = self.fake_B[:,1:2,:,:]
        self.fake_B_blue = self.fake_B[:,2:3,:,:]

        self.real_B_red = self.real_B[:,0:1,:,:]
        self.real_B_green = self.real_B[:,1:2,:,:]
        self.real_B_blue = self.real_B[:,2:3,:,:]

        # color loss, pixel loss, cycle loss, identity loss
        self.loss_G_L1 = (self.criterionL1(self.fake_B_red, self.real_B_red) + self.criterionL1(self.fake_B_green, self.real_B_green) + self.criterionL1(self.fake_B_blue, self.real_B_blue)) * self.opt.lambda_L1 + self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1 + self.criterionL1(self.recovery_A, self.real_A) * self.opt.cyc_L1 + self.criterionL1(self.identity_A, self.real_A) * self.opt.lambda_identity
        # feature loss
        self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B) * self.opt.lambda_feat
        # tv loss
        self.loss_reg = self.opt.REGULARIZATION * (torch.sum(torch.abs(self.fake_B[:, :, :, :-1] - self.fake_B[:, :, :, 1:])) + torch.sum(torch.abs(self.fake_B[:, :, :-1, :] - self.fake_B[:, :, 1:, :])))
        # Combined loss of G
        self.loss_G = self.loss_G_GAN_D1 + self.loss_G_GAN_D2 + self.loss_G_L1 + self.loss_G_VGG + self.loss_reg 

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad([self.netD1, self.netD2], True)
        self.optimizer_D.zero_grad()
        self.backward_D1()
        self.backward_D2()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad([self.netD1, self.netD2], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
