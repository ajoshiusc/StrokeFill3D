"""Flow objective with small, late-time clean-endpoint structural terms."""

import torch


def masked_mean(value, mask):
    return (value.float() * mask).sum() / mask.sum().clamp_min(1)


def training_loss(velocity, velocity_target, sample, target, mask, t, cfg):
    flow = masked_mean((velocity.float() - velocity_target).square(), mask)
    # Endpoint estimates at very noisy times can average plausible anatomy. Limit
    # auxiliary reconstruction pressure to late interpolation times.
    tv = t.view(-1, 1, 1, 1, 1)
    endpoint = torch.where(
        mask > 0, sample.float() + (1 - tv) * velocity.float(), target
    )
    support = mask * (tv >= cfg.get("structural_min_time", 0.5))
    image = masked_mean((endpoint - target).abs(), support)
    gradients = []
    for axis in (2, 3, 4):
        a = [slice(None)] * 5
        b = a.copy()
        a[axis] = slice(1, None)
        b[axis] = slice(None, -1)
        a, b = tuple(a), tuple(b)
        weight = torch.maximum(support[a], support[b])
        gradients.append(
            masked_mean(
                ((endpoint[a] - endpoint[b]) - (target[a] - target[b])).abs(), weight
            )
        )
    gradient = torch.stack(gradients).mean()
    total = (
        flow
        + cfg.get("endpoint_l1_weight", 0) * image
        + cfg.get("gradient_weight", 0) * gradient
    )
    return total, {
        "flow_mse": float(flow.detach()),
        "endpoint_l1": float(image.detach()),
        "gradient_l1": float(gradient.detach()),
    }
