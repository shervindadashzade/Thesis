#%%
import os
import torch.autograd as autograd
from itertools import chain
from collections import OrderedDict, defaultdict
import torch
from torch import nn
from torch.nn import functional as F
from typing import TypeVar, List
from abc import abstractmethod
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Any
import pickle
#%%
def eval_dsnae_epoch(model, data_loader, device, history):
    """

    :param model:
    :param data_loader:
    :param device:
    :param history:
    :return:
    """
    model.eval()
    avg_loss_dict = defaultdict(float)
    for x_batch in data_loader:
        x_batch = x_batch[0].to(device)
        with torch.no_grad():
            loss_dict = model.loss_function(*(model(x_batch)))
            for k, v in loss_dict.items():
                avg_loss_dict[k] += v.cpu().detach().item() / len(data_loader)

    for k, v in avg_loss_dict.items():
        history[k].append(v)
    return history


def dsn_ae_train_step(s_dsnae, t_dsnae, s_batch, t_batch, device, optimizer, history, scheduler=None):
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    s_dsnae.train()
    t_dsnae.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    s_loss_dict = s_dsnae.loss_function(*s_dsnae(s_x))
    t_loss_dict = t_dsnae.loss_function(*t_dsnae(t_x))

    optimizer.zero_grad()
    loss = s_loss_dict['loss'] + t_loss_dict['loss']
    loss.backward()

    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}

    for k, v in loss_dict.items():
        history[k].append(v)

    return history
#%%
Tensor = TypeVar('torch.tensor')
#%%
class MLP(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List = None, dop: float = 0.1, act_fn=nn.SELU, out_fn=None, **kwargs) -> None:
        super(MLP, self).__init__()
        self.output_dim = output_dim
        self.dop = dop

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        modules = []

        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0], bias=True),
                #nn.BatchNorm1d(hidden_dims[0]),
                act_fn(),
                nn.Dropout(self.dop)
            )
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    #nn.BatchNorm1d(hidden_dims[i + 1]),
                    act_fn(),
                    nn.Dropout(self.dop)
                )
            )

        self.module = nn.Sequential(*modules)

        if out_fn is None:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dims[-1], output_dim, bias=True)
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dims[-1], output_dim, bias=True),
                out_fn()
            )



    def forward(self, input: Tensor) -> Tensor:
        embed = self.module(input)
        output = self.output_layer(embed)

        return output
#%%
class BaseAE(nn.Module):
    def __init__(self) -> None:
        super(BaseAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass
#%%
class DSNAE(BaseAE):

    def __init__(self, shared_encoder, decoder, input_dim: int, latent_dim: int, alpha: float = 1.0,
                 hidden_dims: List = None, dop: float = 0.1, noise_flag: bool = False, norm_flag: bool = False,
                 **kwargs) -> None:
        super(DSNAE, self).__init__()
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.noise_flag = noise_flag
        self.dop = dop
        self.norm_flag = norm_flag

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.shared_encoder = shared_encoder
        self.decoder = decoder
        # build encoder
        modules = []

        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0], bias=True),
                # nn.BatchNorm1d(hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(self.dop)
            )
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    # nn.Dropout(0.1),
                    # nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(self.dop)
                )
            )
        modules.append(nn.Dropout(self.dop))
        modules.append(nn.Linear(hidden_dims[-1], latent_dim, bias=True))
        # modules.append(nn.LayerNorm(latent_dim, eps=1e-12, elementwise_affine=False))

        self.private_encoder = nn.Sequential(*modules)

        # build decoder
        # modules = []
        #
        # modules.append(
        #     nn.Sequential(
        #         nn.Linear(2 * latent_dim, hidden_dims[-1], bias=True),
        #         # nn.Dropout(0.1),
        #         nn.BatchNorm1d(hidden_dims[-1]),
        #         nn.ReLU()
        #     )
        # )
        #
        # hidden_dims.reverse()
        #
        # for i in range(len(hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
        #             nn.BatchNorm1d(hidden_dims[i + 1]),
        #             # nn.Dropout(0.1),
        #             nn.ReLU()
        #         )
        #     )
        # self.decoder = nn.Sequential(*modules)

        # self.final_layer = nn.Sequential(
        #     nn.Linear(hidden_dims[-1], hidden_dims[-1], bias=True),
        #     nn.BatchNorm1d(hidden_dims[-1]),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(hidden_dims[-1], input_dim)
        # )

    def p_encode(self, input: Tensor) -> Tensor:
        if self.noise_flag and self.training:
            latent_code = self.private_encoder(input + torch.randn_like(input, requires_grad=False) * 0.1)
        else:
            latent_code = self.private_encoder(input)

        if self.norm_flag:
            return F.normalize(latent_code, p=2, dim=1)
        else:
            return latent_code

    def s_encode(self, input: Tensor) -> Tensor:
        if self.noise_flag and self.training:
            latent_code = self.shared_encoder(input + torch.randn_like(input, requires_grad=False) * 0.1)
        else:
            latent_code = self.shared_encoder(input)
        if self.norm_flag:
            return F.normalize(latent_code, p=2, dim=1)
        else:
            return latent_code

    def encode(self, input: Tensor) -> Tensor:
        p_latent_code = self.p_encode(input)
        s_latent_code = self.s_encode(input)

        return torch.cat((p_latent_code, s_latent_code), dim=1)

    def decode(self, z: Tensor) -> Tensor:
        # embed = self.decoder(z)
        # outputs = self.final_layer(embed)
        outputs = self.decoder(z)

        return outputs

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        z = self.encode(input)
        return [input, self.decode(z), z]

    def loss_function(self, *args, **kwargs) -> dict:
        input = args[0]
        recons = args[1]
        z = args[2]

        p_z = z[:, :z.shape[1] // 2]
        s_z = z[:, z.shape[1] // 2:]

        recons_loss = F.mse_loss(input, recons)

        s_l2_norm = torch.norm(s_z, p=2, dim=1, keepdim=True).detach()
        s_l2 = s_z.div(s_l2_norm.expand_as(s_z) + 1e-6)

        p_l2_norm = torch.norm(p_z, p=2, dim=1, keepdim=True).detach()
        p_l2 = p_z.div(p_l2_norm.expand_as(p_z) + 1e-6)

        ortho_loss = torch.mean((s_l2.t().mm(p_l2)).pow(2))
        # ortho_loss = torch.square(torch.norm(torch.matmul(s_z.t(), p_z), p='fro'))
        # ortho_loss = torch.mean(torch.square(torch.diagonal(torch.matmul(p_z, s_z.t()))))
        # if recons_loss > ortho_loss:
        #     loss = recons_loss + self.alpha * 0.1 * ortho_loss
        # else:
        loss = recons_loss + self.alpha * ortho_loss
        return {'loss': loss, 'recons_loss': recons_loss, 'ortho_loss': ortho_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)

        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[1]
#%%
def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.shape[0], 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    critic_interpolates = critic(interpolates)
    fakes = torch.ones((real_samples.shape[0], 1)).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=fakes,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def critic_dsn_train_step(critic, s_dsnae, t_dsnae, s_batch, t_batch, device, optimizer, history, scheduler=None,
                          clip=None, gp=None):
    critic.zero_grad()
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    s_dsnae.eval()
    t_dsnae.eval()
    critic.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    s_code = s_dsnae.encode(s_x)
    t_code = t_dsnae.encode(t_x)

    loss = torch.mean(critic(t_code)) - torch.mean(critic(s_code))

    if gp is not None:
        gradient_penalty = compute_gradient_penalty(critic,
                                                    real_samples=s_code,
                                                    fake_samples=t_code,
                                                    device=device)
        loss = loss + gp * gradient_penalty

    optimizer.zero_grad()
    loss.backward()
    #     if clip is not None:
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    if clip is not None:
        for p in critic.parameters():
            p.data.clamp_(-clip, clip)
    if scheduler is not None:
        scheduler.step()

    history['critic_loss'].append(loss.cpu().detach().item())

    return history


def gan_dsn_gen_train_step(critic, s_dsnae, t_dsnae, s_batch, t_batch, device, optimizer, alpha, history,
                           scheduler=None):
    critic.zero_grad()
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    critic.eval()
    s_dsnae.train()
    t_dsnae.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    t_code = t_dsnae.encode(t_x)

    optimizer.zero_grad()
    gen_loss = -torch.mean(critic(t_code))
    s_loss_dict = s_dsnae.loss_function(*s_dsnae(s_x))
    t_loss_dict = t_dsnae.loss_function(*t_dsnae(t_x))
    recons_loss = s_loss_dict['loss'] + t_loss_dict['loss']
    loss = recons_loss + alpha * gen_loss
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}

    for k, v in loss_dict.items():
        history[k].append(v)
    history['gen_loss'].append(gen_loss.cpu().detach().item())

    return history


def train_code_adv(s_dataloaders, t_dataloaders, **kwargs):
    """

    :param s_dataloaders:
    :param t_dataloaders:
    :param kwargs:
    :return:
    """
    s_train_dataloader = s_dataloaders[0]
    s_test_dataloader = s_dataloaders[1]

    t_train_dataloader = t_dataloaders[0]
    t_test_dataloader = t_dataloaders[1]

    shared_encoder = MLP(input_dim=kwargs['input_dim'],
                         output_dim=kwargs['latent_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'],
                         dop=kwargs['dop']).to(kwargs['device'])

    shared_decoder = MLP(input_dim=2 * kwargs['latent_dim'],
                         output_dim=kwargs['input_dim'],
                         hidden_dims=kwargs['encoder_hidden_dims'][::-1],
                         dop=kwargs['dop']).to(kwargs['device'])

    s_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    alpha=kwargs['alpha'],
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop'],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])

    t_dsnae = DSNAE(shared_encoder=shared_encoder,
                    decoder=shared_decoder,
                    alpha=kwargs['alpha'],
                    input_dim=kwargs['input_dim'],
                    latent_dim=kwargs['latent_dim'],
                    hidden_dims=kwargs['encoder_hidden_dims'],
                    dop=kwargs['dop'],
                    norm_flag=kwargs['norm_flag']).to(kwargs['device'])

    confounding_classifier = MLP(input_dim=kwargs['latent_dim'] * 2,
                                 output_dim=1,
                                 hidden_dims=kwargs['classifier_hidden_dims'],
                                 dop=kwargs['dop']).to(kwargs['device'])

    ae_params = [t_dsnae.private_encoder.parameters(),
                 s_dsnae.private_encoder.parameters(),
                 shared_decoder.parameters(),
                 shared_encoder.parameters()
                 ]
    t_ae_params = [t_dsnae.private_encoder.parameters(),
                   s_dsnae.private_encoder.parameters(),
                   shared_decoder.parameters(),
                   shared_encoder.parameters()
                   ]

    ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=kwargs['lr'])
    classifier_optimizer = torch.optim.RMSprop(confounding_classifier.parameters(), lr=kwargs['lr'])
    t_ae_optimizer = torch.optim.RMSprop(chain(*t_ae_params), lr=kwargs['lr'])

    dsnae_train_history = defaultdict(list)
    dsnae_val_history = defaultdict(list)
    critic_train_history = defaultdict(list)
    gen_train_history = defaultdict(list)
    # classification_eval_test_history = defaultdict(list)
    # classification_eval_train_history = defaultdict(list)

    # start dsnae pre-training
    for epoch in range(int(kwargs['pretrain_num_epochs'])):
        if epoch % 50 == 0:
            print(f'AE training epoch {epoch}')
        for step, s_batch in enumerate(s_train_dataloader):
            t_batch = next(iter(t_train_dataloader))
            dsnae_train_history = dsn_ae_train_step(s_dsnae=s_dsnae,
                                                    t_dsnae=t_dsnae,
                                                    s_batch=s_batch,
                                                    t_batch=t_batch,
                                                    device=kwargs['device'],
                                                    optimizer=ae_optimizer,
                                                    history=dsnae_train_history)
        dsnae_val_history = eval_dsnae_epoch(model=s_dsnae,
                                                data_loader=s_test_dataloader,
                                                device=kwargs['device'],
                                                history=dsnae_val_history
                                                )
        dsnae_val_history = eval_dsnae_epoch(model=t_dsnae,
                                                data_loader=t_test_dataloader,
                                                device=kwargs['device'],
                                                history=dsnae_val_history
                                                )
        for k in dsnae_val_history:
            if k != 'best_index':
                dsnae_val_history[k][-2] += dsnae_val_history[k][-1]
                dsnae_val_history[k].pop()

    # start critic pre-training
    # for epoch in range(100):
    #     if epoch % 10 == 0:
    #         print(f'confounder critic pre-training epoch {epoch}')
    #     for step, t_batch in enumerate(s_train_dataloader):
    #         s_batch = next(iter(t_train_dataloader))
    #         critic_train_history = critic_dsn_train_step(critic=confounding_classifier,
    #                                                      s_dsnae=s_dsnae,
    #                                                      t_dsnae=t_dsnae,
    #                                                      s_batch=s_batch,
    #                                                      t_batch=t_batch,
    #                                                      device=kwargs['device'],
    #                                                      optimizer=classifier_optimizer,
    #                                                      history=critic_train_history,
    #                                                      clip=None,
    #                                                      gp=None)
    # start GAN training
    for epoch in range(int(kwargs['train_num_epochs'])):
        if epoch % 50 == 0:
            print(f'confounder wgan training epoch {epoch}')
        for step, t_batch in enumerate(t_train_dataloader):
            s_batch = next(iter(s_train_dataloader))
            if s_batch[0].shape[0] != t_batch[0].shape[0]:
                continue
            critic_train_history = critic_dsn_train_step(critic=confounding_classifier,
                                                            s_dsnae=s_dsnae,
                                                            t_dsnae=t_dsnae,
                                                            s_batch=s_batch,
                                                            t_batch=t_batch,
                                                            device=kwargs['device'],
                                                            optimizer=classifier_optimizer,
                                                            history=critic_train_history,
                                                            # clip=0.1,
                                                            gp=10.0)
            if (step + 1) % 5 == 0:
                gen_train_history = gan_dsn_gen_train_step(critic=confounding_classifier,
                                                            s_dsnae=s_dsnae,
                                                            t_dsnae=t_dsnae,
                                                            s_batch=s_batch,
                                                            t_batch=t_batch,
                                                            device=kwargs['device'],
                                                            optimizer=t_ae_optimizer,
                                                            alpha=1.0,
                                                            history=gen_train_history)


    return t_dsnae.shared_encoder, (dsnae_train_history, dsnae_val_history, critic_train_history, gen_train_history)
# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
with open('storage/gene_expressions_cleaned/tcga.pkl','rb') as f:
    tcga_data = pickle.load(f)
with open('storage/gene_expressions_cleaned/gdsc.pkl','rb') as f:
    gdsc_data = pickle.load(f)

tcga_log_transformed = np.log1p(tcga_data['data'])
gdsc_log_transformed = np.log1p(gdsc_data['data'])

tcga_log_transformed = scaler.fit_transform(tcga_log_transformed)
gdsc_log_transformed = scaler.fit_transform(gdsc_log_transformed)

tcga_log_transformed = torch.tensor(tcga_log_transformed, dtype=torch.float32)
gdsc_log_transformed = torch.tensor(gdsc_log_transformed, dtype=torch.float32)

train_tcga, test_tcga = train_test_split(tcga_log_transformed, test_size=0.1, shuffle=True, random_state=42)
train_gdsc, test_gdsc = train_test_split(gdsc_log_transformed, test_size=0.1, shuffle=True, random_state=42)

train_tcga_dataset = torch.utils.data.TensorDataset(train_tcga)
test_tcga_dataset = torch.utils.data.TensorDataset(test_tcga)
train_gdsc_dataset = torch.utils.data.TensorDataset(train_gdsc)
test_gdsc_dataset = torch.utils.data.TensorDataset(test_gdsc)

BATCH_SIZE= 64
lr=0.0001
pretrain_num_epochs=500
train_num_epochs=1000
alpha=1.0
classifier_hidden_dims=[64,32]
latent_dim=128
encoder_hidden_dims=[512,256]
dop=0
input_dim = tcga_log_transformed.shape[1]
device='cpu'
norm_flag=True

train_tcga_loader = torch.utils.data.DataLoader(train_tcga_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_tcga_loader = torch.utils.data.DataLoader(test_tcga_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_gdsc_loader = torch.utils.data.DataLoader(train_gdsc_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_gdsc_loader = torch.utils.data.DataLoader(test_gdsc_dataset, batch_size=BATCH_SIZE, shuffle=True)
#%%
shared_encoder, history = train_code_adv([train_gdsc_loader, test_gdsc_loader],[train_tcga_loader, test_tcga_loader],
                                         input_dim=input_dim,
                                         device=device,
                                         norm_flag=norm_flag,
                                         lr=lr, 
                                         pretrain_num_epochs=pretrain_num_epochs,
                                         train_num_epochs=train_num_epochs,
                                         alpha=alpha,
                                         classifier_hidden_dims=classifier_hidden_dims,
                                         latent_dim=latent_dim,
                                         encoder_hidden_dims=encoder_hidden_dims,
                                         dop=dop)
# %%
torch.save(shared_encoder.state_dict(),'experiments/CODEAE/z_score_normalized/shared_encoder.pt')
with open('experiments/CODEAE/z_score_normalized/history.pkl', 'wb') as f:
    pickle.dump(history, f)
# %%
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
tsne = TSNE(n_components=2, perplexity=30)
shared_encoder.eval()
all_data = torch.concat([test_tcga, test_gdsc],dim=0)
#%%
colors = ['tab:blue' if i< test_tcga.shape[0] else 'tab:orange' for i in range(all_data.shape[0])]
with torch.no_grad():
    out = shared_encoder(all_data).numpy()
tsne_embeddings = tsne.fit_transform(out)
#%%
plt.scatter(tsne_embeddings[:,0],tsne_embeddings[:,1], s=3, c=colors, alpha=0.3)
#%%
# plt.plot(history[0]['loss'],label='dsnae_train_loss')
# plt.plot(history[1]['loss'], label='dsnae_val_loss')
plt.plot(history[2]['critic_loss'], label='critic_train_loss')
plt.plot(history[3]['loss'],label='gen_train_loss')
plt.legend()
#%%
gdsc_data['patient_ids']
#%%
