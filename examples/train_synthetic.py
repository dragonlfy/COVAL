"""CPU smoke test on generated tensors. No clinical data are used."""

import torch

from coval import COVAL, ConceptMap, predict_proba, train_step


def main() -> None:
    torch.manual_seed(42)
    patients, concepts, summaries_per_concept, points = 96, 4, 2, 6
    summary = torch.randn(patients, concepts * summaries_per_concept)
    observed = (torch.rand(patients, concepts) > 0.2).float()
    clock = torch.randn(patients, points, concepts) * observed[:, None, :]
    targets = torch.bernoulli(
        torch.sigmoid(0.8 * summary[:, 0] - 0.5 * summary[:, 3] + 0.4 * clock[:, :, 2].mean(1))
    )
    inputs = {"summary": summary, "clock": clock, "observed": observed}
    concept_map = ConceptMap(
        summary_to_concept=torch.arange(concepts).repeat_interleave(summaries_per_concept),
        trajectory_to_concept=torch.arange(concepts),
        missing_summary=torch.zeros(summary.shape[1]),
        num_concepts=concepts,
    )

    # For this synthetic example only, all inputs already lie on a common
    # scale. In a real study, fit normalization and bin edges on sources only.
    edges = torch.linspace(-4.0, 4.0, 9)
    model = COVAL(
        [edges.clone() for _ in range(summary.shape[1])],
        [edges.clone() for _ in range(concepts)],
        embedding_dim=4,
        k=4,
        d_block=32,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
    for epoch in range(1, 4):
        losses = []
        for ids in torch.randperm(patients).split(32):
            batch = {key: value[ids] for key, value in inputs.items()}
            losses.append(
                train_step(
                    model, batch, targets[ids], optimizer, concept_map,
                    rho=0.25, alpha=0.5, warmup=epoch <= 2,
                )
            )
        # Paired training has two view losses, so divide by two only for
        # comparable display; the optimization objective is unchanged.
        display_loss = sum(losses) / len(losses) / (1 if epoch <= 2 else 2)
        print(f"epoch {epoch}: mean per-view loss = {display_loss:.4f}")
    print("first five probabilities:", predict_proba(model, {k: v[:5] for k, v in inputs.items()}).tolist())


if __name__ == "__main__":
    main()
