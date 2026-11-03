"""Residual-quantized autoencoder; no latent sampler or KL term is needed.

All squared losses use the mean over batch and feature dimensions. Auxiliary
losses are summed across levels. Residuals keep the encoder gradient but detach
earlier selected embeddings, so each codebook is trained by its own loss.
"""
import torch
from torch import nn
from torch.nn import functional as F


def mlp(dimensions):
    layers = []
    for index, (left, right) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(dimensions) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class RQVAE(nn.Module):
    def __init__(self, dimensions=(768, 512, 256, 128, 32), levels=3, codebook_size=256, beta=0.25):
        super().__init__()
        self.encoder = mlp(dimensions)
        self.decoder = mlp(list(reversed(dimensions)))
        self.codebooks = nn.ModuleList(nn.Embedding(codebook_size, dimensions[-1]) for _ in range(levels))
        self.beta = beta

    def quantize(self, latent):
        residual, selected, codes = latent, [], []
        codebook_loss = latent.new_zeros(())
        commitment_loss = latent.new_zeros(())
        for book in self.codebooks:
            distances = residual.square().sum(1, keepdim=True) + book.weight.square().sum(1)[None, :] - 2 * residual @ book.weight.T
            indices = distances.argmin(dim=1)
            vectors = book(indices)
            codebook_loss = codebook_loss + F.mse_loss(vectors, residual.detach())
            commitment_loss = commitment_loss + F.mse_loss(residual, vectors.detach())
            selected.append(vectors)
            codes.append(indices)
            residual = residual - vectors.detach()
        return torch.stack(selected).sum(0), torch.stack(codes, dim=1), codebook_loss, commitment_loss

    def forward(self, values):
        latent = self.encoder(values)
        quantized, codes, codebook_loss, commitment_loss = self.quantize(latent)
        straight_through = latent + (quantized - latent).detach()
        reconstructed = self.decoder(straight_through)
        reconstruction = F.mse_loss(reconstructed, values)
        loss = reconstruction + codebook_loss + self.beta * commitment_loss
        return {"loss": loss, "reconstruction": reconstruction, "codebook": codebook_loss,
                "commitment": commitment_loss, "codes": codes, "latent": latent,
                "reconstructed": reconstructed}

    @torch.no_grad()
    def initialize_codebooks(self, values, seed):
        from sklearn.cluster import KMeans
        if len(values) < self.codebooks[0].num_embeddings:
            raise ValueError("K-means initialization needs at least codebook_size items")
        residual = self.encoder(values).cpu().numpy()
        for level, book in enumerate(self.codebooks):
            fit = KMeans(n_clusters=book.num_embeddings, random_state=seed + level, n_init=10).fit(residual)
            book.weight.copy_(torch.from_numpy(fit.cluster_centers_).to(book.weight))
            residual = residual - fit.cluster_centers_[fit.labels_]


def model_from_settings(settings):
    return RQVAE(**{key: settings[key] for key in ("dimensions", "levels", "codebook_size", "beta")})


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = model_from_settings(checkpoint["settings"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def inspect_model(model, values, batch_size):
    model.eval()
    totals = {key: 0.0 for key in ("loss", "reconstruction", "codebook", "commitment")}
    codes = []
    for batch in values.split(batch_size):
        output = model(batch)
        for key in totals:
            totals[key] += output[key].item() * len(batch)
        codes.append(output["codes"].cpu())
    totals = {key: value / len(values) for key, value in totals.items()}
    codes = torch.cat(codes).numpy()
    import numpy as np
    unique, counts = np.unique(codes, axis=0, return_counts=True)
    totals.update(active_codes=[len(np.unique(codes[:, level])) for level in range(codes.shape[1])],
                  collision_rate=float(1 - len(unique) / len(codes)), largest_collision_group=int(counts.max()))
    return totals, codes
