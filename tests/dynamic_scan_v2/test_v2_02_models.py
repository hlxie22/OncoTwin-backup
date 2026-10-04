from __future__ import annotations

import numpy as np
import torch

from metrics import fractional_censor_nll_per_row
from v2_02_models import (
    PROFILE_TO_CODE,
    PairedStructuredSurvivalModel,
    StructuredBatch,
    apply_observation_profiles,
    fractional_censor_nll_torch,
)


def make_batch(n: int = 5) -> StructuredBatch:
    def block(d):
        return torch.randn(n, d), torch.ones(n, d)
    h, hm = block(13); u, um = block(3); c, cm = block(8); d, dm = block(22)
    oh, ohm = block(10); r, rm = block(10); oc, ocm = block(16); s, sm = block(8)
    return StructuredBatch(
        h, hm, u, um, c, cm, d, dm, oh, ohm, r, rm, oc, ocm, s, sm,
        torch.randn(n, 128), torch.ones(n), torch.rand(n),
    )


def test_fractional_torch_matches_numpy():
    torch.manual_seed(7)
    logits = torch.randn(8, 24, 4)
    time = torch.tensor([1.0, 30.0, 31.0, 45.0, 100.0, 365.0, 729.0, 730.0])
    cause = torch.tensor([0, 1, 0, 2, 3, 0, 1, 0])
    got = fractional_censor_nll_torch(logits, time, cause).detach().numpy()
    expected = fractional_censor_nll_per_row(logits.numpy(), time.numpy(), cause.numpy())
    assert np.allclose(got, expected, atol=2e-5)


def test_portable_profile_removes_optional_before_encoder():
    b = make_batch(3)
    codes = torch.tensor([
        PROFILE_TO_CODE["full_supported"],
        PROFILE_TO_CODE["portable_scan_core"],
        PROFILE_TO_CODE["scan_core_no_genomics"],
    ])
    out = apply_observation_profiles(b, codes)
    assert torch.count_nonzero(out.optional_history[1]) == 0
    assert torch.count_nonzero(out.optional_current[1]) == 0
    assert torch.count_nonzero(out.optional_history_mask[1]) == 0
    assert torch.count_nonzero(out.optional_current_mask[1]) == 0
    assert torch.all(out.stream_availability[1, 2:8] == 0)
    assert torch.all(out.stream_availability_mask[1, 2:8] == 1)
    assert torch.all(out.tumor[2] == 0)
    assert out.genomic_available[2].item() == 0


def test_current_perturbation_cannot_change_pre_logits():
    torch.manual_seed(3)
    b = make_batch(6)
    dims = {
        "history_core": 13, "utilization_core": 3, "current_core": 8, "change_core": 22,
        "optional_history": 10, "optional_history_recency": 10,
        "optional_current": 16, "stream_availability": 8,
    }
    model = PairedStructuredSurvivalModel(include_optional=True, include_genomics=True, dims=dims).eval()
    with torch.no_grad():
        pre1, post1 = model(b)
        changed = b.clone()
        changed.current_core = changed.current_core + 4.0
        changed.change_core = changed.change_core - 2.0
        changed.optional_current = changed.optional_current + 3.0
        pre2, post2 = model(changed)
    assert torch.equal(pre1, pre2)
    assert not torch.equal(post1, post2)


def test_no_genomics_adversarial_invariant():
    torch.manual_seed(9)
    b = make_batch(4)
    codes = torch.full((4,), PROFILE_TO_CODE["no_genomics"], dtype=torch.long)
    a = apply_observation_profiles(b, codes)
    b2 = b.clone(); b2.tumor += 999.0; b2.genomic_age_scaled += 999.0
    c = apply_observation_profiles(b2, codes)
    assert torch.equal(a.tumor, c.tumor)
    assert torch.equal(a.genomic_available, c.genomic_available)
    assert torch.equal(a.genomic_age_scaled, c.genomic_age_scaled)
