import unittest
import torch
import torch.nn.functional as F
from core import coordinate_maps, healthy_sample, inpaint, fit_source


class ConstantEndpoint(torch.nn.Module):
    def forward(self, x, timesteps):
        t = timesteps[:, None, None, None] / 1000
        return (0.2 - x[:, :7]) / (1 - t)


class BlockCodec:
    def encode(self, image):
        return F.avg_pool2d(image, 4).repeat(1, 7, 1, 1)

    def decode(self, code):
        return code[:, :1].repeat_interleave(4, 2).repeat_interleave(4, 3)


class BiasedEncoder(BlockCodec):
    def encode(self, image):
        return super().encode(image) * 0.5


class IdentityPrior(torch.nn.Module):
    def forward(self, x, timesteps):
        return x[:, :7] * 0


class SamplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.coords = torch.zeros(2, 3, 16, 16)

    def test_exact_straight_flow_and_data_projection(self):
        mask = torch.zeros(2, 1, 64, 64)
        mask[:, :, 24:40, 24:40] = 1
        image = torch.full_like(mask, 0.8)
        image[mask > 0] = 0
        for steps in [1, 8, 40]:
            p = inpaint(
                ConstantEndpoint(), BlockCodec(), image, mask, self.coords, steps=steps
            )
            self.assertTrue(torch.equal(p[mask == 0], image[mask == 0]))
            self.assertTrue(
                torch.allclose(
                    p[mask > 0], torch.full_like(p[mask > 0], 0.2), atol=2e-6
                )
            )

    def test_hidden_image_values_are_rejected(self):
        image = torch.ones(2, 1, 64, 64)
        mask = torch.ones_like(image)
        with self.assertRaisesRegex(ValueError, "zero under"):
            inpaint(ConstantEndpoint(), BlockCodec(), image, mask, self.coords)

    def test_no_observations_reduces_to_learned_prior(self):
        code = healthy_sample(ConstantEndpoint(), self.coords, steps=12)
        self.assertTrue(torch.allclose(code, torch.full_like(code, 0.2), atol=2e-6))
        image = torch.zeros(2, 1, 64, 64)
        mask = torch.ones_like(image)
        p = inpaint(
            ConstantEndpoint(), BlockCodec(), image, mask, self.coords, steps=12
        )
        self.assertTrue(torch.allclose(p, torch.full_like(p, 0.2), atol=2e-6))

    def test_roundtrip_bias_does_not_modify_an_unobserved_image(self):
        image = torch.zeros(2, 1, 64, 64)
        mask = torch.ones_like(image)
        p = inpaint(
            ConstantEndpoint(), BiasedEncoder(), image, mask, self.coords, steps=12
        )
        self.assertTrue(torch.allclose(p, torch.full_like(p, 0.2), atol=2e-6))
        old = inpaint(
            ConstantEndpoint(),
            BiasedEncoder(),
            image,
            mask,
            self.coords,
            steps=12,
            projection_mode="roundtrip",
        )
        self.assertGreater(float((old - p).abs().max()), 0.05)

    def test_world_coordinates_match_full_resolution_pooling(self):
        starts = torch.tensor([[12.0, -68.0, 90.0], [76.0, 40.0, 38.0]])
        a = torch.arange(64)
        xx, zz = torch.meshgrid(a, a, indexing="ij")
        full = torch.stack(
            [
                (starts[:, 0, None, None] + xx - 98) / 100,
                starts[:, 1, None, None].expand(-1, 64, 64) / 100,
                (starts[:, 2, None, None] + zz - 72) / 100,
            ],
            dim=1,
        )
        self.assertTrue(
            torch.allclose(coordinate_maps(starts), F.avg_pool2d(full, 4), atol=2e-7)
        )

    def test_source_fit_uses_raw_visible_measurements(self):
        mask = torch.zeros(2, 1, 64, 64)
        mask[:, :, 24:40, 24:40] = 1
        image = torch.full_like(mask, 0.6)
        image[mask > 0] = 0
        p, code, history = fit_source(
            IdentityPrior(),
            BlockCodec(),
            image,
            mask,
            self.coords,
            steps=2,
            iterations=80,
            learning_rate=0.1,
            trust_weight=0.0001,
        )
        self.assertLess(history[-1]["observed_mse"], history[0]["observed_mse"] * 0.02)
        self.assertTrue(torch.equal(p[mask == 0], image[mask == 0]))
        self.assertTrue(torch.isfinite(code).all())


if __name__ == "__main__":
    unittest.main()
