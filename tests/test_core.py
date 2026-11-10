import unittest

import torch

from coval import COVAL, ConceptMap, coval_loss, remove_concepts, train_step


class CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.inputs = {
            "summary": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "clock": torch.arange(12.0).reshape(2, 2, 3),
            "observed": torch.ones(2, 3),
        }
        self.mapping = ConceptMap(
            summary_to_concept=torch.tensor([0, 0, 1]),
            trajectory_to_concept=torch.tensor([0, 1, 2]),
            missing_summary=torch.tensor([-1.0, -2.0, -3.0]),
            num_concepts=3,
        )

    def test_removal_is_synchronized_and_non_mutating(self) -> None:
        drop = torch.tensor([[True, False, False], [False, True, False]])
        reduced = remove_concepts(self.inputs, drop, self.mapping)
        torch.testing.assert_close(reduced["summary"], torch.tensor([[-1.0, -2.0, 3.0], [4.0, 5.0, -3.0]]))
        self.assertTrue((reduced["clock"][0, :, 0] == 0).all())
        self.assertTrue((reduced["clock"][1, :, 1] == 0).all())
        self.assertEqual(reduced["observed"][0, 0].item(), 0.0)
        self.assertEqual(reduced["observed"][1, 1].item(), 0.0)
        self.assertEqual(self.inputs["summary"][0, 0].item(), 1.0)

    def test_guidance_is_detached(self) -> None:
        original = torch.randn(2, 3, 1, requires_grad=True)
        reduced = torch.randn(2, 3, 1, requires_grad=True)
        targets = torch.tensor([0.0, 1.0])
        coval_loss(original, reduced, targets, alpha=0.5).backward()
        hard_only = torch.nn.functional.binary_cross_entropy_with_logits(
            original.detach()[..., 0], targets[:, None].expand(2, 3)
        )
        expected = torch.autograd.grad(
            torch.nn.functional.binary_cross_entropy_with_logits(
                original[..., 0], targets[:, None].expand(2, 3)
            ), original, retain_graph=False
        )[0]
        self.assertTrue(torch.isfinite(hard_only))
        torch.testing.assert_close(original.grad, expected)
        self.assertIsNotNone(reduced.grad)

    def test_forward_and_train_step(self) -> None:
        bins = [torch.linspace(-4, 8, 9) for _ in range(3)]
        model = COVAL(bins, bins, embedding_dim=4, k=3, d_block=16)
        logits = model(self.inputs)
        self.assertEqual(logits.shape, (2, 3, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = train_step(model, self.inputs, torch.tensor([0.0, 1.0]), optimizer, self.mapping)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))


if __name__ == "__main__":
    unittest.main()
