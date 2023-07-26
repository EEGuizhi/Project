import torch

class HeatmapMaker():
    def __init__(self, image_size:tuple, heatmap_std:float):
        self.image_size = image_size
        self.heatmap_std = heatmap_std

    def make_gaussian_heatmap(self, mean:torch.Tensor, size:tuple):
        # mean = coord: (68, 2)
        mean = mean.unsqueeze(1).unsqueeze(1)
        grid = torch.stack(torch.meshgrid(torch.arange(size[0]), torch.arange(size[1])), dim=-1).unsqueeze(0).to(mean.device)
        x_minus_mean = grid - mean

        var = self.heatmap_std ** 2
        gaus = (-0.5 * (x_minus_mean.pow(2) / var)).sum(-1).exp()
        return gaus

    def coord2heatmap(self, coord:torch.Tensor):
        # coord : (batch, 68, 2)
        with torch.no_grad():
            heatmap = torch.stack([
                self.make_gaussian_heatmap(coord_item, size=self.image_size) for coord_item in coord
            ])
        return heatmap

    def heatmap2sargmax_coord(self, heatmap:torch.Tensor):
        # heatmap: (batch, 68, 512, 256) = (batch, ch, row, col)
        heatmap_col = torch.sum(heatmap, dim=-2)  # (batch, ch, col)
        heatmap_row = torch.sum(heatmap, dim=-1)  # (batch, ch, row)
        mesh_c = torch.arange(heatmap_col.shape[-1]).unsqueeze(0).unsqueeze(0).to(heatmap.device)  # (1, 1, col)
        mesh_r = torch.arange(heatmap_row.shape[-1]).unsqueeze(0).unsqueeze(0).to(heatmap.device)  # (1, 1, row)

        coord_c = torch.sum(heatmap_col * mesh_c, dim=-1) / torch.sum(heatmap_col, dim=-1)
        coord_r = torch.sum(heatmap_row * mesh_r, dim=-1) / torch.sum(heatmap_row, dim=-1)
        coord = torch.stack([coord_r, coord_c], dim=-1)  # (batch, 68, 2)

        return coord
