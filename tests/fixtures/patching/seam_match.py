"""A correctly shaped fake satisfies an installer seam: ty accepts it."""

from spindle.backends.miles_runtime.targets import install_multi_target_hooks


class FakeLosses:
    def _target_logprobs(self, args, batch, logits): ...

    def _response_masks(self, batch, log_probs): ...

    def get_log_probs_and_entropy(
        self,
        logits,
        *,
        args,
        unconcat_tokens,
        total_lengths,
        response_lengths,
        with_entropy=False,
        entropy_requires_grad=True,
        non_loss_data=True,
        max_seq_lens=None,
        rollout_sampling_mask=None,
    ): ...


install_multi_target_hooks(FakeLosses())
