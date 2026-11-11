"""Exercise the native verl optimizer contract with uneven context counts."""


def test_native_grpo_counts_rollouts_not_contexts_and_token_loss_masks_gradients():
    import numpy as np
    import torch

    from verl.protocol import DataProto
    from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
    from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories

    # One successful rollout has three contexts; seven failures have one each.
    # Padding is a separate zero-mask UID, never the ninth member of the group.
    keys = ["question_0_0", "question_0_1", "question_0_2"] + [f"question_{i}_0" for i in range(1, 8)] + ["pad123_0_0"]
    masks = torch.tensor([[1, 0, 1]] * 10 + [[0, 0, 0]])
    rewards = torch.zeros(11, 3)
    rewards[:3, -1] = 1.0
    data = DataProto.from_dict(
        tensors={"token_level_rewards": rewards, "response_mask": masks},
        non_tensors={"uid": np.array(["question"] * 10 + ["pad123"], dtype=object)},
    )
    result = compute_advantage_for_multi_trajectories(
        data, keys, AdvantageEstimator.GRPO, num_repeat=8, norm_adv_by_std_in_grpo=False
    )
    expected = torch.tensor([0.875] * 3 + [-0.125] * 7 + [0.0]).unsqueeze(1) * masks
    torch.testing.assert_close(result.batch["advantages"], expected)
    loss_values = torch.arange(33, dtype=torch.float32).reshape(11, 3).requires_grad_()
    loss = agg_loss(loss_values, masks, "token-mean")
    loss.backward()
    assert torch.count_nonzero(loss_values.grad[masks == 0]) == 0
    assert torch.count_nonzero(loss_values.grad[masks == 1]) == 20
    altered = loss_values.detach().masked_fill(masks == 0, 1e9)
    torch.testing.assert_close(agg_loss(altered, masks, "token-mean"), loss.detach())
    # Distributed microbatches use the global valid-token count, not row count.
    distributed = (
        sum(
            agg_loss(loss_values[a:b], masks[a:b], "token-mean", dp_size=2, batch_num_tokens=20)
            for a, b in [(0, 4), (4, 8), (8, 11)]
        )
        / 2
    )
    torch.testing.assert_close(distributed, loss)


def test_native_token_importance_correction_does_not_reactivate_masked_tokens():
    import torch

    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_weights

    ratios = torch.tensor([[0.0, 100.0, 2.0], [0.0, 10.0, 0.0]])
    mask = torch.tensor([[1, 0, 1], [0, 0, 0]])
    weights, _ = compute_rollout_correction_weights(ratios, mask, "token", 2.0)
    torch.testing.assert_close(weights, torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0]]))
    assert not weights.requires_grad
