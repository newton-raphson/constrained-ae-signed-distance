import torch
import torch.nn as nn
class VAEEncoder(nn.Module):
    def __init__(self, input_size, hidden_layer_size=[16,14,12], latent_dim=16):
        super(VAEEncoder, self).__init__()
 
        # Encoder
        self.encoder_layers = nn.ModuleList()
        flat_input_size = input_size[1] * input_size[2]
        current_input_size = flat_input_size
        print("input size")
        print(flat_input_size)
        for hidden_size in hidden_layer_size:
            self.encoder_layers.append(nn.Linear(current_input_size, hidden_size))
            self.encoder_layers.append(nn.ReLU())
            current_input_size = hidden_size
 
        # Outputs for mu and log_var
        self.fc_mu = nn.Linear(current_input_size, latent_dim)
        self.fc_log_var = nn.Linear(current_input_size, latent_dim)
 
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # Standard deviation
        eps = torch.randn_like(std)  # 'randn_like' ensures that the random numbers are of the same type and device as std
        # if not self.training:
        #     return mu+std
        return mu +  eps*std  # Return the reparameterized samples

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        for layer in self.encoder_layers:
            x = layer(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
 
class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, output_size):
        super(VAEDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=8, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=8, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Adjust output to match the range of signed distance field values
        )
 
    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)  # Reshape for the convolutional layers
        return self.decoder(z)
 
class VAE(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=[16,14,12], latent_dim=16, output_size=128):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(input_size, hidden_layer_sizes, latent_dim)
        self.decoder = VAEDecoder(latent_dim, output_size)
 
    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var

# class EncoderDecoder(nn.Module):
#     def __init__(self, input_size, hidden_layer_size=[16,14,12], latent_dim=16,output_size=128):
#         super(EncoderDecoder, self).__init__()

#         # Encoder
#         encoder_layers = []
#         num_hidden_layers = len(hidden_layer_size)
#         flat_input_size = input_size[0] * input_size[1]
#         encoder_layers.append(nn.Linear(flat_input_size, hidden_layer_size[0]))
#         encoder_layers.append(nn.ReLU())
#         for i in range(1,num_hidden_layers):
#             encoder_layers.append(nn.Linear(hidden_layer_size[i-1], hidden_layer_size[i]))
#             encoder_layers.append(nn.ReLU())

#         encoder_layers.append(nn.Linear(hidden_layer_size[-1], latent_dim))
#         encoder_layers.append(nn.ReLU())

#         self.encoder = nn.Sequential(*encoder_layers)

#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(latent_dim, 64, kernel_size=8, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, kernel_size=8, stride=2, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # Adjust output_padding
#             nn.BatchNorm2d(1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # Adjust output_padding
#             nn.Tanh()  # Assuming you want values in the range [0, 1]
#         )

#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten input for the fully connected layer
#         x = self.encoder(x)
#         x = x.view(x.size(0), -1, 1, 1)  # Reshape for the convolutional layers
#         x = self.decoder(x)
#         return x

# class Encoder(nn.Module):
#     def __init__(self, input_size, hidden_layer_size=[16,14,12], latent_dim=16):
#         super(Encoder, self).__init__()

#         # Encoder
#         encoder_layers = []
#         num_hidden_layers = len(hidden_layer_size)
#         flat_input_size = input_size[0] * input_size[1]
#         encoder_layers.append(nn.Linear(flat_input_size, hidden_layer_size[0]))
#         encoder_layers.append(nn.ReLU())
#         for i in range(1,num_hidden_layers):
#             encoder_layers.append(nn.Linear(hidden_layer_size[i-1], hidden_layer_size[i]))
#             encoder_layers.append(nn.ReLU())

#         encoder_layers.append(nn.Linear(hidden_layer_size[-1], latent_dim))
#         encoder_layers.append(nn.ReLU())

#         self.encoder = nn.Sequential(*encoder_layers)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten input for the fully connected layer
#         x = self.encoder(x)
#         return x

# class Decoder(nn.Module):
#     def __init__(self, latent_dim=16,output_size=128):
#         super(Decoder, self).__init__()

#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(latent_dim, 64, kernel_size=8, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, kernel_size=8, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # Adjust output_padding
#             nn.ReLU(),
#             nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # Adjust output_padding
#             nn.Tanh()  # Assuming you want values in the range [0, 1]
#         )

#     def forward(self, x):
#         x = x.view(x.size(0), -1, 1, 1)  # Reshape for the convolutional layers
#         x = self.decoder(x)
#         return x

class Decoder(nn.Module):
    def __init__(self, input_size=16, hidden_layer_size=[16,14,12], latent_dim=16,output_size=128):
        super(Decoder, self).__init__()

        # Encoder
        encoder_layers = []
        num_hidden_layers = len(hidden_layer_size)
        # flat_input_size = input_size[0] * input_size[1]
        encoder_layers.append(nn.Linear(input_size, hidden_layer_size[0]))
        encoder_layers.append(nn.ReLU())
        for i in range(1,num_hidden_layers):
            encoder_layers.append(nn.Linear(hidden_layer_size[i-1], hidden_layer_size[i]))
            encoder_layers.append(nn.ReLU())

        encoder_layers.append(nn.Linear(hidden_layer_size[-1], latent_dim))
        encoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=8, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=8, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # Adjust output_padding
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # Adjust output_padding
            nn.Tanh()  # Assuming you want values in the range [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1, 1, 1)  # Reshape for the convolutional layers
        x = self.decoder(x)
        return x
# Define the decoder architecture
class Encoder(nn.Module):
    def __init__(self, input,latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16,1,kernel_size=1, stride=1, padding=1)
        )
        self.linear = nn.Linear(36,16)
        self.tan = nn.Tanh()


    def forward(self, x):
        z =  self.encoder(x)
        # print(z.shape)
        z =  z.flatten(1)
        # print(z.shape)
        return self.linear(self.tan(z))



class ConstrainedAutoEncoder(nn.Module):
    def __init__(self, input_size=128, hidden_layer_size=[16,14,12], latent_dim=16,output_size=128):
        super(ConstrainedAutoEncoder, self).__init__()

        # Encoder

        self.encoder = Encoder(input_size,latent_dim)
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        # print(z.shape)
        y = self.decoder(z)
        # print(x.shape)
        return y,z