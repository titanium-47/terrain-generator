import torch

def grad_map(height_maps: torch.Tensor, scale: int = 1):
    """
    Computes the gradient maps for a batch of height maps.

    Args:
        height_maps (torch.Tensor): A tensor of shape (n, y, x) representing a batch of height maps.
        scale (int): Represents how far apart to sample the values for the gradient.
    Returns:
        torch.Tensor: A tensor of shape (n, y, x, 2) representing the gradient maps for each height map.
                      The last dimension contains the gradients in the x and y directions.
    """
    n, y, x = height_maps.shape
    grad_x = torch.zeros_like(height_maps)
    grad_y = torch.zeros_like(height_maps)

    grad_y[:, scale:-scale, :] = (height_maps[:, scale*2:, :] - height_maps[:, :-scale*2, :])/(scale*2.0)
    grad_x[:, :, scale:-scale] = (height_maps[:, :, scale*2:] - height_maps[:, :, :-scale*2])/(scale*2.0)

    grad_maps = torch.stack((grad_x, grad_y), dim=-1)
    return grad_maps

def normal_map(height_maps: torch.Tensor, scale: int = 1):
    """
    Computes the normal maps for a batch of height maps.

    Args:
        height_maps (torch.Tensor): A tensor of shape (n, y, x) representing a batch of height maps.
        scale (int): Represents how far apart to sample the values for the gradient.
    Returns:
        torch.Tensor: A tensor of shape (n, y, x, 3) representing the normal maps for each height map.
                      The last dimension contains the normal vectors in the x, y, and z directions.
    """
    n, y, x = height_maps.shape
    grad_x = torch.zeros_like(height_maps)
    grad_y = torch.zeros_like(height_maps)

    grad_y[:, scale:-scale, :] = (height_maps[:, scale*2:, :] - height_maps[:, :-scale*2, :])/(scale*2.0)
    grad_x[:, :, scale:-scale] = (height_maps[:, :, scale*2:] - height_maps[:, :, :-scale*2])/(scale*2.0)

    normal_x = -grad_x
    normal_y = -grad_y
    normal_z = torch.ones_like(height_maps)

    normals = torch.stack((normal_x, normal_y, normal_z), dim=-1)
    norm = torch.norm(normals, dim=-1, keepdim=True)
    normal_maps = normals / norm
    return normal_maps

def second_derivatives_map(height_maps: torch.Tensor, scale: int = 1):
    """
    Computes the second derivative maps for a batch of height maps with respect to x and y.

    Args:
        height_maps (torch.Tensor): A tensor of shape (n, y, x) representing a batch of height maps.
        scale (int): Represents the scale for computing the second derivatives.
    Returns:
        torch.Tensor: A tensor of shape (n, y, x, 2) representing the second derivatives in the x and y directions.
    """
    n, y, x = height_maps.shape
    second_x = torch.zeros_like(height_maps)
    second_y = torch.zeros_like(height_maps)

    second_y[:, scale:-scale, :] = (
        height_maps[:, 2*scale:, :] - 2.0 * height_maps[:, scale:-scale, :] + height_maps[:, :-2*scale, :]
    ) / (scale ** 2)
    
    second_x[:, :, scale:-scale] = (
        height_maps[:, :, 2*scale:] - 2.0 * height_maps[:, :, scale:-scale] + height_maps[:, :, :-2*scale]
    ) / (scale ** 2)
    
    second_derivatives = torch.stack((second_x, second_y), dim=-1)
    return second_derivatives

def laplacian_map(height_maps: torch.Tensor, scale: int = 1):
    """
    Computes the Laplacian maps for a batch of height maps.

    Args:
        height_maps (torch.Tensor): A tensor of shape (n, y, x) representing a batch of height maps.
        scale (int): Represents the scale for computing the Laplacian.
    Returns:
        torch.Tensor: A tensor of shape (n, y, x) representing the Laplacian maps for each height map.
    """
    n, y, x = height_maps.shape
    laplacian = torch.zeros_like(height_maps)

    laplacian[:, scale:-scale, scale:-scale] = (
        height_maps[:, :-2*scale, scale:-scale] +
        height_maps[:, 2*scale:, scale:-scale] +
        height_maps[:, scale:-scale, :-2*scale] +
        height_maps[:, scale:-scale, 2*scale:] -
        4.0 * height_maps[:, scale:-scale, scale:-scale]
    ) / (scale ** 2)

    return laplacian