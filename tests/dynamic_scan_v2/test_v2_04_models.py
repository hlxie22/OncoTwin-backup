import numpy as np
import torch

from v2_02_models import StructuredBatch
from v2_04_models import (
    GenomicProjector,
    HybridTemporalScanModel,
    POSTPROG_CLASSES,
    choose_with_simplicity_margin,
    masked_weighted_ce,
)


def batch(n=5):
    def z(d): return torch.zeros(n, d)
    def o(d): return torch.ones(n, d)
    return StructuredBatch(
        history_core=z(13), history_core_mask=o(13),
        utilization_core=z(3), utilization_core_mask=o(3),
        current_core=z(8), current_core_mask=o(8),
        change_core=z(22), change_core_mask=o(22),
        optional_history=z(10), optional_history_mask=o(10),
        optional_history_recency=z(10), optional_history_recency_mask=o(10),
        optional_current=z(16), optional_current_mask=o(16),
        stream_availability=z(8), stream_availability_mask=o(8),
        tumor=torch.randn(n, 128),
        genomic_available=torch.ones(n),
        genomic_age_scaled=torch.ones(n),
    )


def dims():
    return {
        "history_core": 13, "utilization_core": 3, "current_core": 8,
        "change_core": 22, "optional_history": 10,
        "optional_history_recency": 10, "optional_current": 16,
        "stream_availability": 8,
    }


def test_pre_is_invariant_to_current_scan_content():
    torch.manual_seed(1)
    m = HybridTemporalScanModel(representation_mode="post_temporal_plus_explicit_scan", genomic_mode="embedding", dims=dims()).eval()
    b1 = batch()
    b2 = batch()
    # Preserve history/genomics exactly, perturb only current/change/optional-current.
    for name in ["history_core", "history_core_mask", "utilization_core", "utilization_core_mask", "optional_history", "optional_history_mask", "optional_history_recency", "optional_history_recency_mask", "tumor", "genomic_available", "genomic_age_scaled"]:
        setattr(b2, name, getattr(b1, name).clone())
    b2.current_core = torch.randn_like(b2.current_core) * 20
    b2.change_core = torch.randn_like(b2.change_core) * 20
    b2.optional_current = torch.randn_like(b2.optional_current) * 20
    tp = torch.randn(5, 192)
    tq = torch.randn(5, 192)
    simple = torch.randn(5, 3)
    ctx=torch.randn(5,3)
    with torch.no_grad():
        a = m(b1, tp, tq, simple, ctx)
        c = m(b2, tp, tq, simple, ctx)
    assert torch.equal(a.pre_logits, c.pre_logits)
    assert not torch.equal(a.post_logits, c.post_logits)


def test_pre_temporal_explicit_scan_ignores_temporal_post():
    torch.manual_seed(2)
    m = HybridTemporalScanModel(representation_mode="pre_temporal_plus_explicit_scan", genomic_mode="embedding", dims=dims()).eval()
    b = batch()
    tp = torch.randn(5, 192)
    simple = torch.randn(5, 3)
    ctx=torch.randn(5,3)
    with torch.no_grad():
        x = m(b, tp, torch.randn(5, 192), simple, ctx)
        y = m(b, tp, torch.randn(5, 192) * 50, simple, ctx)
    assert torch.equal(x.pre_logits, y.pre_logits)
    assert torch.equal(x.post_logits, y.post_logits)


def test_temporal_post_only_ignores_current_scan_tensor():
    torch.manual_seed(3)
    m = HybridTemporalScanModel(representation_mode="temporal_post_only", genomic_mode="embedding", dims=dims()).eval()
    b1 = batch(); b2 = batch()
    for name in b1.__dataclass_fields__:
        setattr(b2, name, getattr(b1, name).clone())
    b2.current_core = torch.randn_like(b2.current_core) * 100
    b2.change_core = torch.randn_like(b2.change_core) * 100
    b2.optional_current = torch.randn_like(b2.optional_current) * 100
    tp, tq, simple = torch.randn(5,192), torch.randn(5,192), torch.randn(5,3)
    with torch.no_grad():
        ctx=torch.randn(5,3); x=m(b1,tp,tq,simple,ctx); y=m(b2,tp,tq,simple,ctx)
    assert torch.equal(x.pre_logits,y.pre_logits)
    assert torch.equal(x.post_logits,y.post_logits)


def test_no_genomics_is_adversarially_invariant():
    torch.manual_seed(4)
    p = GenomicProjector("none").eval()
    a = p(torch.randn(4,128), torch.ones(4), torch.randn(4,3))
    b = p(torch.randn(4,128)*1000, torch.zeros(4), torch.randn(4,3)*1000)
    assert torch.equal(a,b)
    assert torch.equal(a, torch.zeros_like(a))


def test_simple_summary_unavailable_content_is_masked():
    torch.manual_seed(5)
    p=GenomicProjector("simple_summary").eval()
    tumor=torch.randn(4,128)
    avail=torch.zeros(4)
    age=torch.randn(4)
    a=p(tumor,avail,torch.randn(4,3))
    b=p(tumor*99,avail,torch.randn(4,3)*99)
    assert torch.equal(a,b)


def test_continued_or_censored_semantics_preserved():
    assert POSTPROG_CLASSES == {"NEXT_TREATMENT":0,"CONTINUED_OR_CENSORED":1,"DEATH":2}


def test_masked_weighted_ce_empty_is_zero_and_finite():
    logits=torch.randn(3,3,requires_grad=True)
    loss=masked_weighted_ce(logits,torch.zeros(3,dtype=torch.long),torch.zeros(3,dtype=torch.bool),torch.ones(3))
    assert float(loss)==0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_simplicity_margin_prefers_simpler_near_tie():
    m={"embedding":0.1700,"simple_summary":0.1702,"availability_age":0.1703,"none":0.1704}
    assert choose_with_simplicity_margin(m,["none","availability_age","simple_summary","embedding"],0.0005)=="none"
    assert choose_with_simplicity_margin(m,["none","availability_age","simple_summary","embedding"],0.0001)=="embedding"


def test_genomic_none_removes_availability_age_but_preserves_scan_number_context():
    p=GenomicProjector("none")
    ctx=torch.tensor([[1.5,1.0,2.0],[0.5,0.0,0.0]])
    got=p.context(ctx)
    assert torch.equal(got[:,0],ctx[:,0])
    assert torch.equal(got[:,1:],torch.zeros_like(got[:,1:]))
