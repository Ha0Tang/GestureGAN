import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import itertools

class GestureGANTwoCycleModel(BaseModel):
    def name(self):
        return 'GestureGANTwoCycleModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        # parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
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
        self.loss_names = ['G_GAN_D1',  'G_GAN_D2', 'G_L1', 'G_VGG', 'reg', 'G','D1','D2']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.opt.saveDisk:
            self.visual_names = ['real_A', 'fake_B', 'real_B', 'fake_A']
        else:
            self.visual_names = ['real_A', 'real_D', 'fake_B', 'real_B', 'real_C', 'recovery_A', 'identity_A', 'fake_A', 'recovery_B', 'identity_B']

        # self.visual_names = ['fake_B', 'fake_D']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G','D1','D2']
        else:  # during test time, only load Gs
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
        combine_realA_realD=torch.cat((self.real_A, self.real_D), 1)
        self.fake_B = self.netG(combine_realA_realD)
        combine_fakeB_realC=torch.cat((self.fake_B, self.real_C), 1)
        self.recovery_A = self.netG(combine_fakeB_realC)

        combine_realB_real_C=torch.cat((self.real_B, self.real_C), 1)
        self.fake_A = self.netG(combine_realB_real_C)
        combine_fakeA_realD=torch.cat((self.fake_A, self.real_D), 1)
        self.recovery_B = self.netG(combine_fakeA_realD)


        combine_realA_realC=torch.cat((self.real_A, self.real_C), 1)
        self.identity_A = self.netG(combine_realA_realC)
        combine_realB_realD=torch.cat((self.real_B, self.real_D), 1)
        self.identity_B = self.netG(combine_realB_realD)

    def backward_D1(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        realA_fakeB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_D1_realA_fakeB = self.netD1(realA_fakeB.detach())
        self.loss_D1_realA_fakeB = self.criterionGAN(pred_D1_realA_fakeB, False)

        # Real
        realA_realB = torch.cat((self.real_A, self.real_B), 1)
        pred_D1_realA_realB = self.netD1(realA_realB)
        self.loss_D1_realA_realB = self.criterionGAN(pred_D1_realA_realB, True)

        # Combined loss
        self.loss_D1 = (self.loss_D1_realA_fakeB + self.loss_D1_realA_realB) * 0.5


        realB_fakeA = self.fake_AB_pool.query(torch.cat((self.real_B, self.fake_A), 1))
        pred_D1_realB_fakeA = self.netD1(realB_fakeA.detach())
        self.loss_D1_realB_fakeA = self.criterionGAN(pred_D1_realB_fakeA, False)

        # Combined loss
        self.loss_D1 = (self.loss_D1_realB_fakeA + self.loss_D1_realA_realB) * 0.5 + self.loss_D1

        self.loss_D1.backward()

    def backward_D2(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        realA_realD_fakeB = self.fake_AB_pool.query(torch.cat((self.real_A, self.real_D, self.fake_B), 1))
        pred_D2_realA_realD_fakeB = self.netD2(realA_realD_fakeB.detach())
        self.loss_D2_realA_realD_fakeB = self.criterionGAN(pred_D2_realA_realD_fakeB, False)

        # Real
        realA_realD_realB = torch.cat((self.real_A, self.real_D, self.real_B), 1)
        pred_D2_realA_realD_realB = self.netD2(realA_realD_realB)
        self.loss_D2_realA_realD_realB = self.criterionGAN(pred_D2_realA_realD_realB, True)

        # Combined loss
        self.loss_D2 = (self.loss_D2_realA_realD_fakeB + self.loss_D2_realA_realD_realB) * 0.5

        realB_realC_fakeA = self.fake_AB_pool.query(torch.cat((self.real_B, self.real_C, self.fake_A), 1))
        pred_D2_realB_realC_fakeA = self.netD2(realB_realC_fakeA.detach())
        self.loss_D2_realB_realC_fakeA = self.criterionGAN(pred_D2_realB_realC_fakeA, False)

        # Real
        realB_realC_realA = torch.cat((self.real_B, self.real_C, self.real_A), 1)
        pred_D2_realB_realC_realA = self.netD2(realB_realC_realA)
        self.loss_D2_realB_realC_realA = self.criterionGAN(pred_D2_realB_realC_realA, True)

        # Combined loss
        self.loss_D2 = (self.loss_D2_realB_realC_fakeA + self.loss_D2_realB_realC_realA) * 0.5 + self.loss_D2

        self.loss_D2.backward()


    def backward_G(self):
        # First, G(A) should fake the discriminator
        realA_fakeB = torch.cat((self.real_A, self.fake_B), 1)
        pred_D1_realA_fakeB = self.netD1(realA_fakeB)
        self.loss_G_GAN_D1 = self.criterionGAN(pred_D1_realA_fakeB, True)

        realB_fakeA = torch.cat((self.real_B, self.fake_A), 1)
        pred_D1_realB_fakeA = self.netD1(realB_fakeA)
        self.loss_G_GAN_D1 = self.criterionGAN(pred_D1_realB_fakeA, True) + self.loss_G_GAN_D1      

        realA_realD_fakeB = torch.cat((self.real_A, self.real_D, self.fake_B), 1)
        pred_D2_realA_realD_fakeB = self.netD2(realA_realD_fakeB)
        self.loss_G_GAN_D2 = self.criterionGAN(pred_D2_realA_realD_fakeB, True)

        realB_realC_fakeA = torch.cat((self.real_B, self.real_C, self.fake_A), 1)
        pred_D2_realB_realC_fakeA = self.netD2(realB_realC_fakeA)
        self.loss_G_GAN_D2 = self.criterionGAN(pred_D2_realB_realC_fakeA, True) + self.loss_G_GAN_D2

        self.fake_B_red = self.fake_B[:,0:1,:,:]
        self.fake_B_green = self.fake_B[:,1:2,:,:]
        self.fake_B_blue = self.fake_B[:,2:3,:,:]
        # print(self.fake_A_red.size())
        self.real_B_red = self.real_B[:,0:1,:,:]
        self.real_B_green = self.real_B[:,1:2,:,:]
        self.real_B_blue = self.real_B[:,2:3,:,:]

        self.fake_A_red = self.fake_A[:,0:1,:,:]
        self.fake_A_green = self.fake_A[:,1:2,:,:]
        self.fake_A_blue = self.fake_A[:,2:3,:,:]
        # print(self.fake_A_red.size())
        self.real_A_red = self.real_A[:,0:1,:,:]
        self.real_A_green = self.real_A[:,1:2,:,:]
        self.real_A_blue = self.real_A[:,2:3,:,:]

        # second, G(A)=B
        self.loss_G_L1 = (self.criterionL1(self.fake_B_red, self.real_B_red) + self.criterionL1(self.fake_B_green, self.real_B_green) + self.criterionL1(self.fake_B_blue, self.real_B_blue)) * self.opt.lambda_L1 + self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1 + self.criterionL1(self.recovery_A, self.real_A) * self.opt.cyc_L1 + self.criterionL1(self.identity_A, self.real_A) * self.opt.lambda_identity + (self.criterionL1(self.fake_A_red, self.real_A_red) + self.criterionL1(self.fake_A_green, self.real_A_green) + self.criterionL1(self.fake_A_blue, self.real_A_blue)) * self.opt.lambda_L1 + self.criterionL1(self.fake_A, self.real_A) * self.opt.lambda_L1 + self.criterionL1(self.recovery_B, self.real_B) * self.opt.cyc_L1 + self.criterionL1(self.identity_B, self.real_B) * self.opt.lambda_identity

        self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B) * self.opt.lambda_feat + self.criterionVGG(self.fake_A, self.real_A) * self.opt.lambda_feat

        self.loss_reg = self.opt.REGULARIZATION * (torch.sum(torch.abs(self.fake_B[:, :, :, :-1] - self.fake_B[:, :, :, 1:])) + torch.sum(torch.abs(self.fake_B[:, :, :-1, :] - self.fake_B[:, :, 1:, :]))) + self.opt.REGULARIZATION * (torch.sum(torch.abs(self.fake_A[:, :, :, :-1] - self.fake_A[:, :, :, 1:])) + torch.sum(torch.abs(self.fake_A[:, :, :-1, :] - self.fake_A[:, :, 1:, :])))

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
